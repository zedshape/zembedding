import csv

# Database is a collection of e-sequences
class Database:
    def __init__(self, database):
        self.sequences = []
        self.initialSupport = {}
        self.frequentSecondElements = set()

        for id, eSeqList in enumerate(database):
            newSeq = EventSequence(id)
            eventList = newSeq.processAdding(eSeqList)
            self.sequences.append(newSeq)

            # Add initial support
            for label in eventList:
                if label not in self.initialSupport:
                    self.initialSupport[label] = 0
                self.initialSupport[label] += 1

    def remove(self):
        for idx, seq in enumerate(self.sequences):
            for iidx, evt in enumerate(seq.sequences):
                if evt.label not in self.initialSupport.keys():
                    del self.sequences[idx].sequences[iidx]

    # print function
    def __str__(self):
        rst = []
        for i, eSeq in enumerate(self.sequences):
            rst.append(format("eSeq %d : %s" % (i, eSeq.__str__())))
        return "\n".join(rst)


# Event-interval sequence (e-sequence)
class EventSequence:
    # init function will receive list-parsed sequence and change it into our own structure
    def __init__(self, id):
        self.id = id
        self.sequences = []  # order of event

    def processAdding(self, eSeqList):
        eventList = set()
        for event in eSeqList:
            newInterval = Interval(event[0], event[1], event[2])
            self.sequences.append(newInterval)
            eventList.add(newInterval.label)
        return eventList

    def __repr__(self):
        rst = []
        for event in self.sequences:
            rst.append(event.__str__())
        return "(" + ", ".join(rst) + ")"


# Interval is a triplet composed of stat time, end time, and label
# it will follow lexicographical rule
class Interval:
    def __init__(self, label, start, end):
        self.label = label
        self.start = start
        self.end = end

    # get the whole duration of each event
    def getDuration(self):
        return self.end - self.start

    # for python-supported hash function (for set operations)
    def __hash__(self):
        return hash((self.label, self.start, self.end))

    def __repr__(self):
        return format("(%s, %d, %d)" % (self.label, self.start, self.end))

    # built-in comparing function (for ordering)
    # our ordering is based on start -> end -> label
    def __lt__(self, other):
        if self.start == other.start:
            if self.end == other.end:
                return self.label < other.label
            else:
                return self.end < other.end
        else:
            return self.start < other.start


def getRelation(A, B, constraints):
    relation = None

    gap = constraints["gap"]

    if abs(B.start - A.start) <= 0:
        if abs(B.end - A.end) <= 0:
            relation = "e"
        elif B.end - A.end > 0:
            relation = "s"
    elif abs(B.end - A.end) <= 0 and B.start - A.start > 0:
        relation = "f"
    elif B.start - A.start > 0 and A.end - B.end > 0:
        relation = "c"
    elif A.end - B.start > 0 and B.start - A.start > 0:
        relation = "o"
    elif abs(B.start - A.end) <= 0:
        relation = "m"
    elif B.start - A.end > 0 and (gap == 0 or B.start - A.end < gap):
        relation = "b"

    return relation


# function for preprocessing data into the shape that the algorithm takes
def preprocess(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        your_list = list(reader)

    distinct_events = set()
    new_list = []
    final_list = []
    timelength = {}
    max_index = 0
    for i in your_list:
        new_list.append(i[0].split(" "))

    for i in new_list:
        max_index = max(int(i[0]), max_index)

    for i in range(max_index + 1):
        final_list.append([])

    for i in new_list:
        final_list[int(i[0])].append((str(i[1]), int(i[2]), int(i[3])))
        distinct_events.add(str(i[1]))
        if int(i[0]) not in timelength:
            timelength[int(i[0])] = 0
        timelength[int(i[0])] = max(timelength[int(i[0])], int(i[3]))

    tseq = len(final_list)
    tdis = len(distinct_events)
    tintv = len(new_list)
    aintv = len(new_list) / len(final_list)
    avgtime = sum(timelength.values()) / len(timelength.keys())

    return tseq, tdis, tintv, aintv, avgtime, final_list


def makeConstraints(argv, database):
    constraints = {}
    constraints["minSupPercent"] = float(argv[0]) if (len(argv) > 0) else 0
    constraints["maxSupPercent"] = float(argv[1]) if (len(argv) > 1) else 0
    constraints["minSup"] = constraints["minSupPercent"] * len(database)
    constraints["maxSup"] = constraints["maxSupPercent"] * len(database)
    constraints["gap"] = float(argv[2]) if (len(argv) > 2) else float("inf")
    return constraints

def getEventIntervalSequences(z):
    return z[0] + z[1] + (z[2],)
