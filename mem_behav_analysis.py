#!/usr/bin/python3
import os
import sys
import re
import argparse
import codecs


class Watcher:
    func_call_patt = re.compile(r"^(\d+)\s*([A-z_]+)\((.*)\)$")
    func_ret_patt = re.compile(r"^(\d+)\s*returns: (.+)$")
    inst_patt = re.compile(r"^(\d+)\s*(r|w) @ (.+)$")
    ta = {}
    tf = {}

    #TODO: 完成这几个回调
    @staticmethod
    def malloc(size):
        pass

    @staticmethod
    def calloc():
        pass

    @staticmethod
    def realloc():
        pass

    @staticmethod
    def free(addr):
        pass

    @staticmethod
    def inst_read(addr):
        pass

    @staticmethod
    def inst_write(addr):
        pass

    operations = {
        'free': free,
        'malloc': malloc,
        'calloc': calloc,
        'realloc': realloc,
        'r': inst_read,
        'w': inst_write
    }

    def __init__(self, talloc):
        # 保存原始talloc数据
        self.talloc = talloc
        # 保存talloc数据的解析结果
        self.status = []
        # 记录函数名和参数，在函数返回时与ret一同构造完整函数调用记录
        self.func_call = None

    def handle_op(self, op, *etc):
        # TODO: 完成调用op的操作
        pass

    def watch_line(self, line):
        line = line.strip()
        try:
            self.func_call = self.func_call_patt.findall(line)[0]
            if self.func_call[1] == 'free':
                self.status.append(self.func_call)
                self.func_call = None
        except:
            pass
        try:
            _id, ret = self.func_ret_patt.findall(line)[0]
            if self.func_call is not None:
                _, name, args = self.func_call
                self.status.append((_id, name, args, ret))
                self.func_call = None
        except:
            pass
        try:
            # 避免*alloc内部执行的读写指令误触发报警
            # if self.func_call is not None:
            #     return
            _id, op, addr = self.inst_patt.findall(line)[0]
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
        print(tup)
