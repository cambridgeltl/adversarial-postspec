import sys
reload(sys)
sys.setdefaultencoding('utf8')


def get_line(fh):
    for line in fh:
        yield line

if __name__ == '__main__':

    in_simlex_file = sys.argv[1]
    data = [line.split() for line in open(in_simlex_file).read().splitlines()]
    simlex_list = [iteminner for listinner in data for iteminner in listinner]
    #print >> sys.stderr, simlex_list

    for i, line in enumerate(get_line(sys.stdin)):
        items = line.strip().split()
        counter = 0
        for item in items[1:]:
            if item == '1':
                counter += 1
        if i % 1000 == 0: print >> sys.stderr, 'Processed... ' + i.__str__()

        if items[0] not in simlex_list:
            continue

        print >> sys.stdout, line.strip()


