#!/usr/bin/python3
import os
import sys
import re
import argparse
import codecs
from villoc import Misc, Worker as VWorker
import bisect
from enum import Enum
from rx import create, subject, Observable


class InstStat(Enum):
    OK = 0
    ERR = -1
    UAF = 1
    OVF = 2


class HeapBlock:
    '''
    用于表示堆块
    '''
    header = 8
    footer = 0
    round_sz = 0x10
    min_sz = 0x20

    def __init__(self, addr, size):
        self.uaddr = addr
        self.usize = size

    def get_rsize(self):
        size = max(self.min_sz, self.usize + self.header + self.footer)
        # rsize for "rounded size"
        rsize = size + (self.round_sz - 1)
        rsize = rsize - (rsize % self.round_sz)
        return rsize

    def get_usize(self):
        return self.usize

    def get_uaddr(self):
        return self.uaddr

    def start(self):
        '''返回boundary下界'''
        return self.uaddr - self.header

    def end(self):
        '''返回boundary上界'''
        rsize = self.get_rsize()
        return self.uaddr - self.header + rsize

    def change_rsize(self, new_size: int):
        '''
        改变整个block的size，即self.end() - self.start()的大小
        '''
        self.usize = new_size - self.header - self.footer
        # TODO: 在分配堆块时是否会将空闲堆块切割到小于堆块最小值？

    def __lt__(self, another):
        return self.start() < another.start()

    def __gt__(self, another):
        return self.start() > another.start()

    def __eq__(self, another):
        return self.start() == another.start()

    def __str__(self):
        return "uaddr 0x{:x}, usize {}".format(self.uaddr, self.usize)


class HeapRepr:
    '''
    用于表示内存分配表（其实例为ta、tb等）
    '''

    def __init__(self):
        self.__ta = []
        self.__tf = []

    def get_ta(self):
        return self.__ta[:]

    def get_tf(self):
        return self.__tf[:]

    def __maximized_min_idx(self, lst: list, target_addr: int):
        '''
        对堆块基址的二分查找：
        找到基址小于等于target，且基址最大的堆块下标
        '''
        begin = 0
        end = len(lst)
        # lst中尚无元素
        if begin == end:
            return None
        end -= 1
        while begin < end:
            mid = begin + (end - begin + 1) // 2
            if lst[mid].start() == target_addr:
                return mid
            if lst[mid].start() > target_addr:
                end = mid - 1
            else:
                begin = mid
        if lst[begin].start() > target_addr:
            return None
        return begin

    def is_addr_valid(self, lst: list, addr: int) -> HeapBlock:
        '''
        此处的valid意为给定的addr位于某个block之内
        '''
        pos = self.__maximized_min_idx(lst, addr)
        if pos is None:
            return None
        block: HeapBlock = self.__ta[pos]
        # 若pos非空，即已经隐含“addr >= block.start()”这个条件了。
        # 故这里只需判断addr < block.end()
        if addr < block.end():
            return block

    def __tf_prev_idx_of(self, block: HeapBlock):
        prev_idx = self.__maximized_min_idx(self.__tf, block.start())
        if prev_idx is None:
            return None
        return prev_idx

    def __tf_next_idx_of(self, block: HeapBlock):
        next_idx = self.__maximized_min_idx(self.__tf, block.end())
        if next_idx is None:
            return None
        next_block = self.__tf[next_idx]
        # “后一个堆块”实际上位于当前堆块之前，不需向后合并
        if next_block.begin() != block.end():
            return None
        return next_idx

    def __ta_insert(self, block: HeapBlock):
        bisect.insort_left(self.__ta, block)

    def __ta_pop(self, uaddr: int):
        idx = self.__maximized_min_idx(self.__ta, uaddr)
        if idx is None:
            return None
        if self.__ta[idx].uaddr != uaddr:
            return None
        return self.__ta.pop(idx)

    def __tf_insert(self, block: HeapBlock):
        '''
        在向tf插入元素时需要处理向前/向后合并
        '''
        next_idx = self.__tf_next_idx_of(block)
        prev_idx = self.__tf_prev_idx_of(block)
        next_block: HeapBlock = None if next_idx is None else self.__tf[next_idx]
        prev_block: HeapBlock = None if prev_idx is None else self.__tf[prev_idx]
        if next_block is not None and prev_block is not None:
            next_rsize = next_block.end() - next_block.start()
            curr_rsize = block.end() - block.start()
            prev_rsize = prev_block.end() - prev_block.start()
            prev_block.change_rsize(next_rsize + curr_rsize + prev_rsize)
            self.__tf.pop(next_idx)
        elif next_block is not None:
            next_block.uaddr = block.uaddr
            next_rsize = next_block.end() - next_block.start()
            curr_rsize = block.end() - block.start()
            next_block.change_rsize(next_rsize + curr_rsize)
        elif prev_block is not None:
            prev_rsize = prev_block.end() - prev_block.start()
            curr_rsize = block.end() - block.start()
            prev_block.change_rsize(prev_rsize + curr_rsize)
        else:
            bisect.insort_left(self.__tf, block)

    def __tf_pop(self, block: HeapBlock):
        '''
        在tf中切割已有记录：

        # +----------------+
        # |       fb   +===|=========+
        # +------------+---+  block  |
        #              +=============+
        # 若block.start()位于fb（front_block）中，切割fb尾部；

        #           +----------------+
        # +=========+===+   bb       |
        # |  block  +---+------------+
        # +=============+
        # 若block.end()位于bb（back_block）中，切割bb首部。

        # +------------------+
        # | fb/bb +=======+  |
        # +-------+-------+--+
        #         | block |
        #         +=======+
        # 若fb、bb是同一个对象，则将其分为两半；
        #                       some blocks between fb & bb
        # +-------------+   +--------+     ...       +--------+   +-------------+
        # |    fb   +===+===+========+===============+========+===+===+    bb   |
        # +---------+---+   +--------+    block      +--------+   +---+---------+
        #           +=================================================+
        # 否则就删去它们之间的所有元素。
        '''
        front_block: HeapBlock = self.is_addr_valid(self.__tf, block.start())
        back_block: HeapBlock = self.is_addr_valid(self.__tf, block.end())
        if front_block is not None and back_block is not None:
            if front_block is back_block:
                new_front_block_sz = block.start() - front_block.start()
                new_back_block_sz = back_block.end() - block.end()
                if new_front_block_sz > 0:
                    front_block.change_rsize(new_front_block_sz)
                else:
                    self.__tf.reomve(front_block)
                if new_back_block_sz > 0:
                    new_back_block = HeapBlock(block.end(), 0x10)
                    new_back_block.change_rsize(new_back_block_sz)
                    self.__tf_insert(new_back_block)
            # block的start和end位于不同的fb中，删去这两个tb之间的tb
            else:
                front_block_idx = self.__tf.index(front_block)
                back_block_idx = self.__tf.index(back_block)
                self.__tf = self.__tf[:front_block_idx + 1] + \
                    self.__tf[back_block_idx:]
        elif front_block is not None:
            new_front_block_sz = block.start() - front_block.start()
            if new_front_block_sz > 0:
                front_block.change_rsize(new_front_block_sz)
            else:
                self.__tf.reomve(front_block)
        elif back_block is not None:
            new_back_block_sz = back_block.end() - block.end()
            if new_back_block_sz > 0:
                back_block.change_rsize(new_back_block_sz)
            else:
                self.__tf.remove(back_block)
        elif block in self.__tf:
            self.__tf.remove(block)

    def allocate(self, block: HeapBlock):
        '''
        负责响应内存分配事件：
        ta中添加block；
        检查block是否与tf中的记录重叠，若重叠即进行切割。
        '''
        self.__ta_insert(block)
        self.__tf_pop(block)

    def free(self, uaddr: int):
        '''
        负责响应内存释放事件：
        ta中移除一个元素；
        tf中添加一个元素，添加时处理向前合并、向后合并。
        '''
        freed_block: HeapBlock = self.__ta_pop(uaddr)
        if freed_block is None:
            return
        self.__tf_insert(freed_block)

    def __str__(self):
        return "ta: " + \
            str(list(map(lambda item: str(item), self.__ta))) +\
            "\n" + "tf: " +\
            str(list(map(lambda item: str(item), self.__tf)))


class Watcher:
    func_call_patt = re.compile(r"^(\d+)\s*([A-z_]+)\((.*)\)$")
    func_ret_patt = re.compile(r"^(\d+)\s*returns: (.+)$")
    inst_patt = re.compile(r"^(\d+)\s*(r|w) @ (.+)$")

    def is_uaf(self, addr):
        '''
        addr in ta -> Valid memory access;
        addr not in ta && addr not in tf -> Invalid memory access;
        addr not in ta && addr in tf -> Use-after-free
        '''
        ta = self.heap_repr.get_ta()
        tf = self.heap_repr.get_tf()
        addr_in_ta = self.heap_repr.is_addr_valid(ta, addr)
        addr_in_tf = self.heap_repr.is_addr_valid(tf, addr)
        if not addr_in_ta and addr_in_tf:
            return True
        return False

    def is_heap_overflow(self, addr):
        # TODO: 完成overflow判断方法
        return False

    def malloc(self, size, ret):
        self.heap_repr.allocate(HeapBlock(ret, size))

    def calloc(self, nmemb, size, ret):
        self.malloc(nmemb * size, ret)

    def realloc(self, ptr, size, ret):
        if ptr:
            self.free(ptr)
        self.malloc(size, ret)

    def free(self, addr):
        self.heap_repr.free(addr)

    def inst_read(self, addr):
        is_uaf = self.is_uaf(addr)
        is_overflow = self.is_heap_overflow(addr)
        # TODO: 处理异常事件
        if is_uaf:
            return InstStat.UAF
        if is_overflow:
            return InstStat.OVF
        return InstStat.OK

    def inst_write(self, addr):
        is_uaf = self.is_uaf(addr)
        is_overflow = self.is_heap_overflow(addr)
        # TODO: 处理异常事件
        if is_uaf:
            return InstStat.UAF
        if is_overflow:
            return InstStat.OVF
        return InstStat.OK

    def __init__(self, talloc=None):
        # 保存原始talloc数据
        self.talloc = talloc
        # 保存talloc数据的解析结果
        self.status = []
        # 记录函数名和参数，在函数返回时与ret一同构造完整函数调用记录
        self.func_call = None
        self.heap_repr = HeapRepr()
        self.operations = {
            'free': self.free,
            'malloc': self.malloc,
            'calloc': self.calloc,
            'realloc': self.realloc,
            'r': self.inst_read,
            'w': self.inst_write
        }

    def handle_op(self, op_str, *etc):
        if op_str not in self.operations:
            return
        op = self.operations[op_str]
        return op(*etc)

    def watch_line(self, line):
        # TODO: 返回处理结果
        line = line.strip()

        # 读取函数调用事件
        try:
            self.func_call = self.func_call_patt.findall(line)[0]
            # 由于free没有返回值，故对其单独处理
            if self.func_call[1] == 'free':
                _, op, arg = self.func_call
                self.handle_op(op, Misc.sanitize(arg))
                self.status.append(self.func_call)
                self.func_call = None
                return (op, Misc.sanitize(arg))
            return None
        except Exception as e:
            pass

        # 读取函数返回值，与函数调用一并拼接成完整调用事件
        try:
            _id, ret = self.func_ret_patt.findall(line)[0]
            ret = Misc.sanitize(ret)
            if self.func_call is not None:
                _, op, args = self.func_call
                args = list(
                    map(lambda arg: Misc.sanitize(arg), args.split(',')))
                self.handle_op(op, *args, ret)
                self.status.append((_id, op, *args, ret))
                self.func_call = None
                return (op, *args, ret)
        except Exception as e:
            pass

        # 读取指令执行事件
        try:
            _id, op, addr = self.inst_patt.findall(line)[0]
            addr = Misc.sanitize(addr)
            result = self.handle_op(op, addr)
            self.status.append((_id, op, addr))
            return (op, addr, result)
            # return op
        except Exception as e:
            pass

    def watch(self):
        for line in self.talloc:
            self.watch_line(line)
        print(self.heap_repr)
        return self.status


class Reader:
    '''
    继承该基类，实现对各种数据源的读取功能
    '''

    def has_next(self):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError


class FileReader(Reader):
    def __init__(self, lines):
        self.lines = lines.readlines()
        self.cursor = 0

    def has_next(self):
        return self.cursor < len(self.lines)

    def next(self):
        line = self.lines[self.cursor]
        self.cursor += 1
        return line


class Worker:
    # TODO: 完成该主类
    def __init__(self, reader):
        self.__reader: Reader = reader
        self.__watcher = Watcher()
        # self.__observers = [VWorker(), self]
        self.__observers = [self]
        self.__subject = subject.Subject()

    def start(self):
        self.register()
        while(self.__reader.has_next()):
            line = self.__reader.next()
            try:
                trace = self.__watcher.watch_line(line)
                self.__subject.on_next(trace)
            except Exception as e:
                self.__subject.on_error(e)
        self.__subject.on_completed()

    def register(self):
        for observer in self.__observers:
            observer.subscribe(self.__subject)

    def subscribe(self, observable: Observable):
        observable.subscribe(
            on_next=lambda ev: print(ev),
            # on_error=lambda err: print("err: ", err),
            on_completed=lambda: print("completed")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("talloc", type=argparse.FileType("rb"))
    # parser.add_argument("out", type=argparse.FileType("w"))
    # parser.add_argument("--header", type=int, default=8,
    #                     help="size of malloc metadata before user data")
    # parser.add_argument("--footer", type=int, default=0,
    #                     help="size of malloc metadata after user data")
    # parser.add_argument("--round", type=int, default=0x10,
    #                     help="size of malloc chunks are a multiple of this value")
    # parser.add_argument("--minsz", type=int, default=0x20,
    #                     help="size of a malloc chunk is at least this value")
    # parser.add_argument("--raw", action="store_true",
    #                     help="disables header, footer, round and minsz")

    # # Some values that work well: 38, 917, 190, 226
    # parser.add_argument("-s", "--seed", type=int, default=226)
    # parser.add_argument("-S", "--show-seed", action="store_true")
    args = parser.parse_args()

    noerrors = codecs.getreader('utf8')(args.talloc.detach(), errors='ignore')

    fr = FileReader(noerrors)
    Worker(fr).start()

    # watcher = Watcher(noerrors)
    # for tup in watcher.watch():
    # print(tup)
    # pass
