"""Content comparison of the v1.43 BD with v1.42b, ignoring ordering. Prints every difference; exit 1 if any
difference falls outside seg_stamp_0's module-reference bookkeeping."""
import json
import sys

a = json.load(open('bd_backups/OG_build_v1.42b.bd', encoding='utf-8'))
b = json.load(open('top.srcs/sources_1/bd/OG_build/OG_build.bd', encoding='utf-8'))
out = []


def walk(p, x, y):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in x.keys() | y.keys():
            if k not in x:
                out.append(('ADDED', p + '/' + k, json.dumps(y[k])[:90]))
            elif k not in y:
                out.append(('REMOVED', p + '/' + k, json.dumps(x[k])[:90]))
            else:
                walk(p + '/' + k, x[k], y[k])
    elif isinstance(x, list) and isinstance(y, list):
        if sorted(map(json.dumps, x)) != sorted(map(json.dumps, y)):
            out.append(('LIST', p, ''))
    elif x != y:
        out.append(('CHANGED', p, repr(x)[:50] + ' -> ' + repr(y)[:50]))


walk('', a, b)
bad = [o for o in out if not o[1].startswith('/design/components/seg_stamp_0/reference_info')]
for o in sorted(out):
    print(o)
print('differences', len(out), 'outside seg_stamp reference_info', len(bad))
sys.exit(1 if bad else 0)
