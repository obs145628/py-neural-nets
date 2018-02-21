import sys

import shell

class TestSuite:

    def __init__(self):
        self.out = sys.stdout
        self.out_err = open('./errors.log', 'w')
        self.group = None
        self.test = None
        self.group_tests = 0
        self.group_valids = 0
        self.total_tests = 0
        self.total_valids = 0

    def begin(self):
        return


    def end(self):
        per = self.total_valids / self.total_tests * 100
        self.out.write("Global results: {} / {} ({:.2f}%)\n".format(self.total_valids , self.total_tests, per))

    def begin_group(self, name):
        self.group = name
        self.out.write("Running test suite " + self.group + '\n')
        self.group_tests = 0
        self.group_valids = 0

    def end_group(self):
        per = self.group_valids / self.group_tests * 100
        self.out.write("Test suite {}: {} / {} ({:.2f}%)\n\n".format(self.group, self.group_valids, self.group_tests, per))
        self.out.flush()

        self.total_tests += self.group_tests
        self.total_valids += self.group_valids

        self.group = None

    def begin_test(self, name):
        self.test = name
        self.group_tests += 1

        self.out.write(self.test + "    ")
        self.out.flush()

    def end_test(self, valid, msg = "", msg_more = ""):
        
        if valid:
            self.group_valids += 1
            self.out.write(shell.COLOR_GREEN + "[OK]\n" + shell.COLOR_DEFAULT)
        else:
            self.out.write(shell.COLOR_RED + "[KO]\n")

        self.out.write(msg)
        self.out.write(shell.COLOR_DEFAULT)
                
        self.out_err.write(self.group + '.' + self.test + ': [' + str(valid) + '] \n')
        self.out_err.write(msg)
        self.out_err.write(msg_more)
        self.out_err.write('\n\n')
