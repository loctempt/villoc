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

    def binary_serach(self, lst: list, target, begin=None, end=None):
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

    def is_block_exists(self, table: dict, addr: int):
        lst = sorted(list(table))
        pos = self.binary_serach(lst, addr)
        if pos is None:
            return False
        base = lst[pos]
        return addr < base + table[base] - 1

    def is_uaf(self, addr):
        # TODO: 完成uaf判断方法
        pass

    def is_heap_overflow(self, addr):
        # TODO: 完成overflow判断方法
        pass

    #TODO: 完成这几个回调
    def malloc(self, size, ret):
        self.ta[ret] = size
        if ret not in self.tf:
            return
        # TODO: 切割tf记录
        # print("malloc", size, "@", ret)

    def calloc(self, nmemb, size, ret):
        self.malloc(nmemb * size, ret)
        # self.ta[ret] = nmemb * size
        # print("calloc", nmemb, size, ret)
        # print("calloc {} @ {}".format(nmemb * size, ret))

    def realloc(self, ptr, size, ret):
        if ptr:
            self.free(ptr)
        self.malloc(size, ret)
        # print("realloc", ptr, size, ret)
        # print("realloc {} @ {}".format())

    def free(self, addr):
        if addr not in self.ta:
            return
        self.tf[addr] = self.ta[addr]
        self.ta.pop(addr)
        # print("free", addr)

    def inst_read(self, addr):
        # TODO: uaf overflow 分别编写方法
        if self.is_block_exists(self.ta, addr):
            return InstStat.OK
        if self.is_block_exists(self.tb, addr):
            pass

    def inst_write(self, addr):
        # TODO: 调用uaf和overflow处理方法
        pass
        # print("write @", addr)

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
