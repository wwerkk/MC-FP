"""Small example OSC server/client.
"""
import argparse
import math

from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server, udp_client

def handle_info(unused_addr, args, info):
  msg = args[0]
  print(f"[{msg}] ~ {info}")
  if info == "ping":
    client.send_message("/info", "pong")

if __name__ == "__main__":
  client = udp_client.SimpleUDPClient("127.0.0.1", 5555)
  dispatcher = Dispatcher()
  dispatcher.map("/info", handle_info, "Info")
  server = osc_server.ThreadingOSCUDPServer(
      ("127.0.0.1", 4444), dispatcher)
  print("Serving on {}".format(server.server_address))
  server.serve_forever()