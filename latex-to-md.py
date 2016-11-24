import argparse, string

parser = argparse.ArgumentParser()
parser.add_argument('--m', type = str, default = "") # number of hid layer
args = parser.parse_args()

output = []

with open(args.m, 'r') as f:
	for line in f.readlines():
		line = list(line)
		l = len(line) - 1
		for i, c in enumerate(line):
			if i == 0:
				if c == '$' and line[i+1] != '$':
					line[i] = '$$'
			elif i == l-1:
				if c == '$' and line[i-1] != '$':
					line[i] = '$$'
			else:
				if c == '$' and line[i-1] != '$' and line[i+1] != '$':
					line[i] = '$$'
		output.append(string.join(line, sep = ''))
f.close()

with open(args.m, 'w') as f:
	for line in output:
		f.write(line)
f.close()
