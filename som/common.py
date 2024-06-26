
import datetime
import os
import subprocess

class Task:
    def __init__(self, *command, stdoutFilename=None, stdoutFancy=True):
        self.command = list(command)
        self.stdout = None
        self.stderr = None
        self.process = None
        self.started = False

        self.stdoutFilename = stdoutFilename
        self.stdoutFancy = stdoutFancy

    def start(self, *extraArgs):
        if self.started:
            return

        os.makedirs('output', exist_ok=True)
        name = str(self)
        self.stdout = open('output/' + name + '.out' if not self.stdoutFilename else self.stdoutFilename, 'ab' if self.stdoutFancy else 'wb', 0)
        self.stderr = open('output/' + name + '.err', 'ab', 0)

        header = '\n' + '*' * 120 + '\n' + ' '.join(self.command) + '\n' + str(datetime.datetime.now()) + '\n'
        if self.stdoutFancy:
            self.stdout.write(header.encode('ascii'))
        self.stderr.write(header.encode('ascii'))

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1' # https://stackoverflow.com/a/52851238

        self.process = subprocess.Popen(self.command + list(extraArgs), env=env,
                                        stdout=self.stdout, stderr=self.stderr)
        self.started = True

    def done(self):
        return self.process is not None and self.process.poll() is not None

    def terminate(self):
        if not self.started:
            return
        if self.process:
            self.process.kill()
        if self.stdout:
            self.stdout.close()
        if self.stderr:
            self.stderr.close()
        self.started = False

    def __del__(self):
        self.terminate()

    def __str__(self):
        name = '_'.join(self.command)
        if '/' in name:
            name = name.rsplit('/', 1)[1]
        return name.strip().replace('.', '-')

    def __repr__(self):
        return 'Task(' + ', '.join("'%s'" % arg for arg in self.command) + \
                     ', stdoutFilename=' + str(self.stdoutFilename)

def readKeggRpairs(filePath='../som9/rpair'):
    if '~' in filePath:
        filePath = os.path.expanduser(filePath)

    rpairs = []
    with open(filePath, 'r') as f:
        rpair = None
        readingEnzymes = False
        readingAlignment = False
        readingKcfLeft = False
        readingKcfRight = False
        for line in f:
            label = line[0:12].rstrip()
            content = line[12:].strip()

            if label == 'ENTRY':
                assert(rpair is None)
                rpair = {'id': content.split()[0]}
                continue

            if label == 'NAME':
                rpair['name'] = content
                continue

            if label == 'ENZYME':
                assert('enzymes' not in rpair)
                readingEnzymes = True
                rpair['enzymes'] = []
            elif label != '':
                readingEnzymes = False

            if readingEnzymes:
                rpair['enzymes'].extend(content.split())
                continue

            if label == 'ALIGN':
                assert(not readingAlignment and 'alignment' not in rpair)
                readingAlignment = True
                rpair['alignment'] = []
                continue # skip alignment indices count
            elif label != '':
                readingAlignment = False

            if readingAlignment:
                fields = content.split()
                left = fields[1]
                right = fields[2]

                tags = []
                for i in range(3, len(fields)):
                    assert(fields[i][0] == '#')
                    tag = fields[i][1:]
                    if tag[0] in ['R', 'D', 'M']:
                        tags.append((tag[0], int(tag[1:])))
                    elif tag == 'nonR':
                        tags.append(tag)

                if left == '*':
                    left = None
                else:
                    atomIndex, atomType = left.split(':')
                    left = (int(atomIndex), atomType)

                if right == '*':
                    right = None
                else:
                    atomIndex, atomType = right.split(':')
                    right = (int(atomIndex), atomType)

                rpair['alignment'].append((left, right, tags))

            if label == 'ENTRY1':
                assert(not readingKcfLeft and 'kcfLeft' not in rpair)
                readingKcfLeft = True
                rpair['kcfLeft'] = ''
                continue # skip empty content
            elif label != '' and not label.startswith('  '):
                readingKcfLeft = False

            if readingKcfLeft:
                assert(label == '' or label[0:2] == '  ')
                rpair['kcfLeft'] += '%-12s%s\n' % (label[2:], content)

            if label == 'ENTRY2':
                assert(not readingKcfRight and 'kcfRight' not in rpair)
                readingKcfRight = True
                rpair['kcfRight'] = ''
                continue # skip empty content
            elif label != '' and not label.startswith('  '):
                readingKcfRight = False

            if readingKcfRight:
                assert(label == '' or label[0:2] == '  ')
                rpair['kcfRight'] += '%-12s%s\n' % (label[2:], content)

            if label == '///':
                assert(rpair is not None)
                assert(all(x in rpair for x in ['id', 'name', 'alignment', 'kcfLeft', 'kcfRight']))

                if 'enzymes' not in rpair:
                    rpair['enzymes'] = []

                rpairs.append(rpair)

                rpair = None
                readingEnzymes = False
                readingAlignment = False
                readingKcfLeft = False
                readingKcfRight = False

    return rpairs
