#!/usr/bin/python3

import numpy as np
import pandas as pd
import pyshark
import sys

def get_deltas(filename, max_session=0):
	cap = pyshark.FileCapture(filename, keep_packets=False)
	sessions = {}

	for pkt in cap:
		if "TCP" not in pkt:
			continue
		stream_id = int(pkt.tcp.stream)
		src = pkt.ip.src

		sys.stdout.write("\rStream ID: {}".format(stream_id))
		sys.stdout.flush()

		if max_session != 0 and stream_id >= max_session+5:
			break

		if max_session != 0 and stream_id >= max_session:
			continue
		
		if stream_id not in sessions:
			sessions[stream_id] = {}

		if "delta_time_a2b" in sessions[stream_id]:
			if src == sessions[stream_id]["host_a"]:
				sessions[stream_id]["delta_time_a2b"].append((float(pkt.sniff_timestamp) - sessions[stream_id]["prev_timestamp_a2b"]) * 1000000)
				sessions[stream_id]["prev_timestamp_a2b"] = float(pkt.sniff_timestamp)
			elif sessions[stream_id]["prev_timestamp_b2a"] == -1:
				sessions[stream_id]["prev_timestamp_b2a"] = float(pkt.sniff_timestamp)
			else:
				sessions[stream_id]["delta_time_b2a"].append((float(pkt.sniff_timestamp) - sessions[stream_id]["prev_timestamp_b2a"]) * 1000000)
				sessions[stream_id]["prev_timestamp_b2a"] = float(pkt.sniff_timestamp)
		else:
			sessions[stream_id]["host_a"] = pkt.ip.src
			sessions[stream_id]["host_b"] = pkt.ip.dst
			sessions[stream_id]["port_a"] = pkt.tcp.srcport
			sessions[stream_id]["port_b"] = pkt.tcp.dstport
			sessions[stream_id]["delta_time_a2b"] = []
			sessions[stream_id]["delta_time_b2a"] = []
			sessions[stream_id]["prev_timestamp_a2b"] = float(pkt.sniff_timestamp)
			sessions[stream_id]["prev_timestamp_b2a"] = -1

	print("finished reading pcap")
	cap.close()
	deltas = []

	for index, stream in sessions.items():
		num_deltas_a2b = len(stream["delta_time_a2b"])
		if num_deltas_a2b == 0:
			avg_delta_a2b = 0
		else:
			avg_delta_a2b = sum(stream["delta_time_a2b"]) / num_deltas_a2b
	
		num_deltas_b2a = len(stream["delta_time_b2a"])
		if num_deltas_b2a == 0:
			avg_delta_b2a = 0
		else:
			avg_delta_b2a = sum(stream["delta_time_b2a"]) / num_deltas_b2a

		deltas.append([stream["host_a"], stream["host_b"], stream["port_a"], stream["port_b"], num_deltas_a2b, avg_delta_a2b, num_deltas_b2a, avg_delta_b2a])

	deltas = np.asarray(deltas)
	return deltas

filename = sys.argv[1]
csv_name = sys.argv[2]

try:
	max_session = int(sys.argv[3])
except IndexError:
	max_session = 0

headers = ["host_a", "host_b", "port_a", "port_b", "num_packets_a2b", "avg_delta_time_a2b", "num_packets_b2a", "avg_delta_time_b2a"]
pd_deltas = pd.DataFrame(get_deltas(filename, max_session))
pd_deltas.to_csv(csv_name, header=headers, index=False)
