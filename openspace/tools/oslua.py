"""Send Lua to a running OpenSpace through its local TCP server (port 4681).

Usage: python oslua.py "openspace.printInfo('hi')" ["more lua" ...]
Each argument is sent as one luascript message and its reply is printed.
The protocol (handshake, topic/type/payload) mirrors openspace-api-js.
"""
import json
import socket
import sys

HANDSHAKE = {"type": "apiHandshake", "apiVersion": {"major": 1, "minor": 0, "patch": 0}}


def send(scripts, host="127.0.0.1", port=4681, timeout=10.0):
    replies = []
    with socket.create_connection((host, port), timeout=timeout) as s:
        f = s.makefile("rwb")
        f.write((json.dumps(HANDSHAKE) + "\n").encode())
        f.flush()
        for topic, script in enumerate(scripts):
            msg = {"topic": topic, "type": "luascript",
                   "payload": {"script": script, "return": True,
                               "shouldBeSynchronized": True}}
            f.write((json.dumps(msg) + "\n").encode())
            f.flush()
            reply = None
            try:
                while True:
                    line = f.readline().decode().strip()
                    if not line:
                        break
                    data = json.loads(line)
                    if data.get("topic") == topic:
                        reply = data.get("payload")
                        break
            except OSError:
                # A timed-out socket file can't be read again, so stop here.
                replies.append("(no reply)")
                print(json.dumps("(no reply)"))
                break
            replies.append(reply)
            print(json.dumps(reply))
    return replies


if __name__ == "__main__":
    send(sys.argv[1:])
