#!/usr/bin/python3
import os
import sys
import re
import argparse
import codecs
from villoc import Misc
import bisect
from enum import Enum


class InstStat(Enum):
    OK = 0
    ERR = -1


class Watcher:
    func_call_patt = re.compile(r"^(\d+)\s*([A-z_]+)\((.*)\)$")
    func_ret_patt = re.compile(r"^(\d+)\s*returns: (.+)$")
    inst_patt = re.compile(r"^(\d+)\s*(r|w) @ (.+)$")
    ta = {}
    tf = {}

    def __binary_serach(self, lst: list, target, begin=None, end=None):
        '''
        对堆块基址的二分查找：
        找到基址小于等于target，且基址最大的堆块下标
        '''
        if begin is None:
            begin = 0
        if end is None:
            end = len(lst)
        if begin >= end:
            raise Exception("Begin is greater than end.")
        end -= 1
        while begin < end:
            mid = begin + (end - begin + 1) // 2
            if lst[mid] == target:
                return mid
            if lst[mid] > target:
                end = mid - 1
            else:
                begin = mid
        if lst[begin] > target:
            return None
        return begin

    def __is_addr_valid(self, table: dict, addr: int):
        lst = sorted(list(table))
        pos = self.__binary_serach(lst, addr)
        if pos is None:
            return False
        base = lst[pos]
        return addr < base + table[base]

    def is_uaf(self, addr):
        # TODO: 完成uaf判断方法
        pass

    def is_heap_overflow(self, addr):
        # TODO: 完成overflow判断方法
        pass

    def split_tb(self, base, new_block_size):
        '''
        检查新分配的堆块“nb(new block)”是否与tf中记录的空闲堆块“tb(block in tf)”存在以下关系：
        nb在tb首部与tb相交但不包含、
        nb在tb尾部与tb相交但不包含、
        nb包含于tb。
        '''
        nb_head = base
        nb_tail = base + new_block_size
        nb_head_in_tb = self.__is_addr_valid(self.tf, nb_head)
        nb_tail_in_tb = self.__is_addr_valid(self.tf, nb_tail)

        # 无需切割
        if not (nb_head_in_tb or nb_tail_in_tb):
            return

        # 对tf进行切割
        lst = sorted(list(self.tf))
        # =============================================
        # 1: nb包含于tb || 2: nb在tb尾部与tb相交但不包含
        #
        # 情况1：
        # +----------------+
        # |  tb  +------+  |
        # +------+------+--+
        #        |  nb  |
        #        +------+
        # 情况2：
        # +----------------+
        # |       tb   +---|---------+
        # +------------+---+  nb     |
        #              +-------------+
        # =============================================
        if nb_head_in_tb:
            pos = self.__binary_serach(lst, nb_head)
            tb_base = lst[pos]
            tb_size = self.tf[tb_base]
            self.tf[tb_base] = nb_head - tb_base
            if nb_tail_in_tb:
                self.tf[nb_tail] = tb_size - (nb_tail - tb_base)
        # =============================================
        # nb在tb首部与tb相交但不包含
        #           +----------------+
        # +---------+---+   tb       |
        # |    nb   +---+------------+
        # +-------------+
        # =============================================
        elif nb_tail_in_tb:
            pos = self.__binary_serach(lst, nb_tail)
            tb_base = lst[pos]
            tb_size = self.tf[tb_base]
            self.tf.pop(tb_base)
            self.tf[nb_tail] = tb_size - (nb_tail - tb_base)

    def malloc(self, size, ret):
        self.ta[ret] = size
        # 新的堆块占据了tf中堆块记录的空间，需要对该记录进行分割
        self.split_tb(ret, size)

    def calloc(self, nmemb, size, ret):
        self.malloc(nmemb * size, ret)

    def realloc(self, ptr, size, ret):
        if ptr:
            self.free(ptr)
        self.malloc(size, ret)

    def free(self, addr):
        if addr not in self.ta:
            return
        self.tf[addr] = self.ta[addr]
        self.ta.pop(addr)

    def inst_read(self, addr):
        # TODO: uaf overflow 分别编写方法
        self.is_uaf(addr)
        self.is_heap_overflow(addr)

    def inst_write(self, addr):
        # TODO: 调用uaf和overflow处理方法
        self.is_uaf(addr)
        self.is_heap_overflow(addr)

    def __init__(self, talloc):
        # 保存原始talloc数据
        self.talloc = talloc
        # 保存talloc数据的解析结果
        self.status = []
        # 记录函数名和参数，在函数返回时与ret一同构造完整函数调用记录
        self.func_call = None
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
        op(*etc)

    def watch_line(self, line):
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
        except:
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
        except:
            pass

        # 读取指令执行事件
        try:
            _id, op, addr = self.inst_patt.findall(line)[0]
            addr = Misc.sanitize(addr)
            self.handle_op(op, addr)
            self.status.append((_id, op, addr))
        except:
            pass

    def watch(self):
        for line in self.talloc:
            self.watch_line(line)
        return self.status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("talloc", type=argparse.FileType("rb"))
    # parser.add_argument("out", type=argparse.FileType("w"))
    args = parser.parse_args()

    noerrors = codecs.getreader('utf8')(args.talloc.detach(), errors='ignore')

    watcher = Watcher(noerrors)
    for tup in watcher.watch():
        # print(tup)
        pass
