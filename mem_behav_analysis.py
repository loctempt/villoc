#!/usr/bin/python3
import os
import sys
import re
import argparse
import codecs
import bisect
from villoc import Misc, Worker as VWorker
from rx import create, subject, Observable
from enumerations import *


class SeqRW:
    '''
    SeqRW for Sequential Read and Write
    用于标记连续的读写指令信息
    '''

    def __init__(self):
        self.__new_seq = True
        self.__type = None
        self.__base = -1
        self.__size = 0

    def __do_update(self, type_, addr, size):
        self.__type = type_
        self.__base = addr
        self.__size = size

    def is_new_seq(self) -> bool:
        return self.__new_seq

    def get_type(self) -> str:
        return self.__type

    def get_base(self) -> int:
        return self.__base

    def get_size(self) -> int:
        return self.__size

    def end(self):
        return self.__base + self.__size

    def update(self, type_, addr, size) -> bool:
        if type_ != self.__type:
            self.__do_update(type_, addr, size)
            self.__new_seq = True
            return True
        if self.__base + self.__size == addr:
            self.__do_update(self.__type, self.__base, self.__size + size)
            self.__new_seq = False
            return False
        self.__do_update(self.__type, addr, size)
        self.__new_seq = True
        return True


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
        return self.__ta

    def get_tf(self):
        return self.__tf

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
        block: HeapBlock = lst[pos]
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
        if next_block.start() != block.end():
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


class UaddrRepr:
    def __init__(self, stat=UaddrStat.INIT):
        self.stat = stat
        self.refs = []

    def appendRef(self, ref):
        self.refs.append(ref)

    def removeRef(self, ref):
        if ref in self.refs:
            self.refs.remove(ref)

class PointerRepr:
    def __init__(self, pointerRepr=None, point_to=None, stat=None):
        if pointerRepr is None and point_to is not None and stat is not None:
            self.point_to = point_to
            self.stat = stat
        elif pointerRepr is not None and point_to is None and stat is None:
            self.point_to = pointerRepr.point_to
            self.stat = pointerRepr.stat
        else:
            print('PointerRepr: init failed')


class TaintTracer:
    def __init__(self):
        self.tainted_regs = {}
        self.uaddr_pool = {}
        self.pointer_pool = {}

    def __wash_reg(self, reg):
        if reg in self.tainted_regs:
            self.tainted_regs.pop(reg)

    def __wash_pointer(self, pointer_addr):
        if pointer_addr not in self.pointer_pool:
            return
        uaddr = self.pointer_pool[pointer_addr].point_to
        if uaddr in self.uaddr_pool:
            self.uaddr_pool[uaddr].removeRef(pointer_addr)
        self.pointer_pool.pop(pointer_addr)

    def __wash(self, pointer_addr):
        if type(pointer_addr) is int:
            self.__wash_pointer(pointer_addr)
        else:
            self.__wash_reg(pointer_addr)

    def __taint_pointer(self, pointer_addr, pointer_repr: PointerRepr):
        pr = PointerRepr(pointer_repr)
        self.pointer_pool[pointer_addr] = pr
        uaddr = pr.point_to
        self.uaddr_pool[uaddr].appendRef(pr)

    def __taint_reg(self, reg, pointer_repr: PointerRepr):
        self.tainted_regs[reg] = pointer_repr

    def __taint(self, addr, pointer_repr: PointerRepr):
        if type(addr) is int:
            self.__taint_pointer(addr, pointer_repr)
        else:
            self.__taint_reg(addr, pointer_repr)

    def __get_reg_taint(self, reg):
        if reg not in self.tainted_regs:
            return None
        return self.tainted_regs[reg]

    def __get_pointer_taint(self, pointer_addr):
        if pointer_addr not in self.pointer_pool:
            return None
        return self.pointer_pool[pointer_addr]

    def __get_taint(self, pointer_addr):
        if type(pointer_addr) is int:
            return self.__get_pointer_taint(pointer_addr)
        else:
            return self.__get_reg_taint(pointer_addr)

    def __is_dangling_reg(self, reg):
        if self.is_reg_tainted(reg):
            return self.tainted_regs[reg].stat == PointerStat.DANGLING
        return False

    def is_dangling_pointer(self, pointer_addr):
        '''
        检查pointer_addr对应的指针是否是悬空指针。
        亦可传入reg，其名称就被视为寄存器的地址。
        '''
        if type(pointer_addr) is not int:
            return self.__is_dangling_reg(pointer_addr)
        if pointer_addr in self.pointer_pool:
            return self.pointer_pool[pointer_addr].stat == PointerStat.DANGLING
        return False

    def is_reg_tainted(self, reg):
        return reg in self.tainted_regs

    def is_init_assignment(self, reg_val):
        '''
        检查reg_val是否是一个uaddr，并且处于INIT状态
        '''
        if reg_val not in self.uaddr_pool:
            return False
        return self.uaddr_pool[reg_val].stat == UaddrStat.INIT

    # 在free时调用
    def on_free(self, uaddr):
        # 防止对同一地址free多次时出错
        if uaddr not in self.uaddr_pool:
            return
        uaddr_repr: UaddrRepr = self.uaddr_pool[uaddr]
        for pointerRepr in uaddr_repr.refs:
            pointerRepr.stat = PointerStat.DANGLING
        self.uaddr_pool.pop(uaddr)

    # 在malloc时调用
    def on_allocate(self, uaddr):
        self.uaddr_pool[uaddr] = UaddrRepr(UaddrStat.INIT)

    def update_reg(self, pointer_addr, reg, reg_val):
        '''
        处理read指令时更新寄存器的污染状态
        '''
        pointer_repr = self.__get_taint(pointer_addr)
        # 指针未被标记，wash目标寄存器污点
        if pointer_repr is None:
            self.__wash(reg)
        # 寄存器传递
        else:
            self.__taint(reg, pointer_repr)

    def update_pointer(self, pointer_addr, reg, reg_val):
        '''
        处理write指令时实现指针状态的传递
        '''
        # initial assignment
        if self.is_init_assignment(reg_val):
            uaddr = reg_val
            uaddr_repr = self.uaddr_pool[uaddr]
            uaddr_repr.stat = UaddrStat.INUSE
            # 这是唯一一个可以解除指针悬挂状态的地方
            pointer_repr = PointerRepr(point_to=uaddr, stat=PointerStat.VALID)
            uaddr_repr.appendRef(pointer_repr)
            self.pointer_pool[pointer_addr] = pointer_repr
        else:
            pointer_repr = self.__get_taint(reg)
            # reg未携带指针状态，wash目标污点
            if pointer_repr is None:
                self.__wash(pointer_addr)
            # 一般传递
            else:
                self.__taint(pointer_addr, pointer_repr)


class Watcher:

    def is_basic_uaf(self, addr):
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

    def is_dangling_uaf(self, pointer_addr, reg, reg_val):
        '''
        该方法只会被inst_write()或inst_call()调用。
        在effective address为悬空指针时，存在以下三种非UAF的例外情况：
        1. write时，reg_val是一个INIT状态的uaddr（初始化）；
        2. write时，reg表示一个有效指针（传递有效指针）；
        3. write时，reg表示一个悬空指针（传递悬空指针）。
        在reg未被污染，且effective address为悬空指针时，表明出现UAF；
        除此以外，均为正常情况。

        type(pointer_addr) is str时表示检查call指令是否出发UAF
        '''
        if type(pointer_addr) is str:
            return self.taint_tracer.is_dangling_pointer(pointer_addr)
        if self.taint_tracer.is_init_assignment(reg_val):
            return False
        if not self.taint_tracer.is_reg_tainted(reg):
            return self.taint_tracer.is_dangling_pointer(pointer_addr)
        return False

    def is_uaf(self, addr, reg=None, reg_val=None):
        '''
        addr为int时，先检查是否产生了基本UAF，若有则直接返回；
        若无，再判断是否因使用了悬空指针而造成UAF。
        如果addr不是int，说明此时addr代表寄存器名，
        此时则直接判断程序是否正在使用悬空指针。
        '''
        if type(addr) is int and self.is_basic_uaf(addr):
            return True
        return self.is_dangling_uaf(addr, reg, reg_val)

    def is_heap_overflow(self, addr, size):
        is_new_seq = self.srw.is_new_seq()
        base_addr = -1
        if is_new_seq:
            base_addr = addr
        else:
            base_addr = self.srw.get_base()
        ta = self.heap_repr.get_ta()
        block = self.heap_repr.is_addr_valid(ta, base_addr)
        if not block:
            return False
        if self.srw.end() > block.end():
            return True
        return False

    def malloc(self, ret, size):
        self.heap_repr.allocate(HeapBlock(ret, size))
        self.taint_tracer.on_allocate(ret)

    def calloc(self, ret, nmemb, size):
        self.malloc(ret, nmemb * size)

    def realloc(self, ret, ptr, size):
        if ptr:
            self.free(ptr)
        self.malloc(ret, size)

    def free(self, addr):
        self.heap_repr.free(addr)
        self.taint_tracer.on_free(addr)

    def inst_read(self, addr, size, reg, reg_val):
        self.srw.update('r', addr, size)
        is_uaf = self.is_basic_uaf(addr)
        self.taint_tracer.update_reg(addr, reg, reg_val)
        if is_uaf:
            return (InstStat.UAF, self.srw.get_base(), self.srw.get_size())
        # TODO： 正常情况下也要返回去抖序列，便于villoc装箱
        return (InstStat.OK,)

    def inst_write(self, addr, size, reg, reg_val):
        '''
        write指令中，如果addr是悬空指针，说明发生了UAF，可直接返回；
        反之，引起指针状态传递。
        '''
        # TODO: 考虑一下序列读写记录是否需要改动
        self.srw.update('w', addr, size)
        is_uaf = self.is_uaf(addr, reg, reg_val)
        is_overflow = self.is_heap_overflow(addr, size)
        self.taint_tracer.update_pointer(addr, reg, reg_val)
        # TODO: 考虑是否可能同时发生多种内存错误
        if is_uaf:
            return (InstStat.UAF, self.srw.get_base(), self.srw.get_size())
        if is_overflow:
            return (InstStat.OVF, self.srw.get_base(), self.srw.get_size())
        # TODO: 正常情况下也要返回去抖序列，便于villoc装箱
        return (InstStat.OK,)

    def inst_call(self, reg):
        '''
        如果call指令执行时reg实际上是一个悬空指针，说明出现了UAF。
        '''
        # 此时传入的第三个参数只是为了占位，以便复用方法调用，没有实际用途
        is_uaf = self.is_uaf(reg, reg, -1)
        if is_uaf:
            return (InstStat.UAF, -1, -1)
        return (InstStat.OK,)

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
            'w': self.inst_write,
            'call': self.inst_call
        }
        self.srw = SeqRW()
        self.critical_depth = None  # 用于标记连续free的深度，去除抖动
        self.critical_value = None  # 用于标记连续free的参数，去除抖动
        self.taint_tracer = TaintTracer()

    def handle_op(self, op_str, *etc):
        if op_str not in self.operations:
            return
        op = self.operations[op_str]
        return op(*etc)

    # e.g. 35463     calloc(0x40, 0x1)
    func_call_patt = re.compile(r"^(\d+)\s*(\d+)\s*([A-z_]+)\((.*)\)$")
    # e.g. 50978     returns: 0x5654ff382260
    func_ret_patt = re.compile(r"^(\d+)\s*(\d+)\s*returns: (.+)$")
    # e.g. 50979     w < 0x7ffda49c0e00 0x8 rax 0x5654ff382260
    #      50980     w @ 0x7ffda49c0df8 0x8 *invalid* 0
    inst_patt = re.compile(r"^(\d+)\s*(r|w) (@|<|>) (.+) (.+) (.+) (.+)$")
    # e.g. 53547     call rdx
    inst_call_patt = re.compile(r"^(\d+)\s*(call) (.+)$")

    def watch_line(self, line):
        line = line.strip()
        # 读取函数调用事件
        func_call = self.func_call_patt.findall(line)
        # 由于free没有返回值，故对其单独处理
        if len(func_call) > 0:
            self.func_call = func_call[0]
            if self.func_call[2] == 'free':
                _, depth, op, arg = self.func_call
                depth = int(depth)
                ret = ("skip",)
                # print((self.critical_depth is not None and self.critical_depth +
                    #    1 == depth and self.critical_value == arg))
                if not(self.critical_depth is not None and self.critical_depth+1 == depth and self.critical_value == arg):
                    self.handle_op(op, Misc.sanitize(arg))
                    ret = (op, 0, Misc.sanitize(arg))
                self.critical_depth, self.critical_value = depth, arg
                self.func_call = None
                return ret
            else:
                return ("skip",)
        # 读取函数返回值，与函数调用一并拼接成完整调用事件
        func_return = self.func_ret_patt.findall(line)
        if len(func_return) > 0:
            _, __, ret = func_return[0]
            ret = Misc.sanitize(ret)
            if self.func_call is not None:
                _, __, op, args = self.func_call
                args = list(
                    map(lambda arg: Misc.sanitize(arg), args.split(',')))
                self.handle_op(op, ret, *args)
                # self.status.append((_id, op, ret, *args))
                self.func_call = None
                return (op, ret, *args)
        # 读取指令执行事件
        inst_exec = self.inst_patt.findall(line)
        if len(inst_exec) > 0:
            _, op, __, addr, size, reg, reg_val = inst_exec[0]
            addr = Misc.sanitize(addr)
            size = Misc.sanitize(size)
            reg = Misc.sanitize(reg)
            reg_val = Misc.sanitize(reg_val)
            result = self.handle_op(op, addr, size, reg, reg_val)
            # self.status.append((_id, op, addr))
            return (op, *result)
        # 读取call指令执行事件
        call_exec = self.inst_call_patt.findall(line)
        if len(call_exec) > 0:
            _, op, reg = call_exec[0]
            op = Misc.sanitize(op)
            reg = Misc.sanitize(reg)
            result = self.handle_op(op, reg)
            return (op, *result)

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
    def __init__(self, talloc):
        lines = codecs.getreader('utf8')(talloc, errors='ignore')
        self.lines = lines.readlines()
        self.cursor = 0

    def has_next(self):
        return self.cursor < len(self.lines)

    def next(self):
        line = self.lines[self.cursor]
        self.cursor += 1
        return line


class Worker:
    def __init__(
        self, reader: Reader,
        out,
        header=8,
        footer=0,
        round_=0x10,
        minsz=0x20,
        raw=False,
        seed=226,
        show_seed=False,
        debug=False
    ):
        self.__reader = reader
        self.__watcher = Watcher()
        self.__subject = subject.Subject()
        if not debug:
            self.__observers = [
                VWorker(
                    out,
                    header,
                    footer,
                    round_,
                    minsz,
                    raw,
                    seed,
                    show_seed
                ),
            ]
            return

        # =========== debug env ==============
        self.__observers = [self]

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
            on_next=lambda trace: print(trace),
            # on_error=lambda trace: print(trace)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("talloc", type=argparse.FileType("rb"))
    parser.add_argument("out", type=argparse.FileType("w"))
    parser.add_argument("--header", type=int, default=8,
                        help="size of malloc metadata before user data")
    parser.add_argument("--footer", type=int, default=0,
                        help="size of malloc metadata after user data")
    parser.add_argument("--round", type=int, default=0x10,
                        help="size of malloc chunks are a multiple of this value")
    parser.add_argument("--minsz", type=int, default=0x20,
                        help="size of a malloc chunk is at least this value")
    parser.add_argument("--raw", action="store_true",
                        help="disables header, footer, round and minsz")
    parser.add_argument("--debug", action="store_true")

    # Some values that work well: 38, 917, 190, 226
    parser.add_argument("-s", "--seed", type=int, default=226)
    parser.add_argument("-S", "--show-seed", action="store_true")
    parser.add_argument("--ia32", action="store_true")  # TODO: 添加一个32位开关方便使用
    args = parser.parse_args()

    # noerrors = codecs.getreader('utf8')(args.talloc.detach(), errors='ignore')

    fr = FileReader(args.talloc.detach())
    Worker(fr,
           args.out,
           args.header, args.footer,
           args.round, args.minsz,
           args.raw, args.seed, args.show_seed,
           args.debug
           ).start()

    # print(args)
    # watcher = Watcher(noerrors)
    # for tup in watcher.watch():
    # print(tup)
    # pass
