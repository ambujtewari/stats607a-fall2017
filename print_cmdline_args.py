import sys

print 'This python\'s script name is ' + sys.argv[0]

if len(sys.argv) > 1:
    print 'It was called with the following arguments:',
    for arg in sys.argv[1:]:
        print arg,
    print
else:
    print 'It was called with no arguments.'
