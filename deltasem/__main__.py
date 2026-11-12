import argparse
import json
from . import DeltaSem, Occurrence

parser = argparse.ArgumentParser(description="Maintain aggregates from a JSONL event trace")
parser.add_argument("trace", help="JSONL: insert/delete/resolve/publish events")
args = parser.parse_args()
engine = DeltaSem(lambda key, text: int(json.loads(text)["label"]))
with open(args.trace, encoding="utf8") as stream:
    for line in stream:
        event = json.loads(line)
        op = event.pop("op")
        if op == "insert":
            engine.insert(Occurrence(**event))
        elif op == "delete":
            engine.delete(event["identity"])
        elif op == "resolve":
            engine.resolve(event["identity"])
        elif op == "publish":
            print(json.dumps(engine.publish(**event)))
        else:
            raise ValueError(f"unknown operation: {op}")
