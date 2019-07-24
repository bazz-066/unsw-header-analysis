#!/usr/bin/python3

from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing, svm, tree, metrics
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy
import sys
import traceback
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
#warnings.filterwarnings("ignore", category=UndefinedMetricWarning) 
plt.style.use('seaborn-whitegrid')

def init_conf(tool):
	print("Reading configuration...")
	conf = {}
	if tool == "tcptrace":
		conf["dataset_root_dir"] = "/media/baskoro/HD-LXU3/Datasets/UNSW/UNSW-NB15-Source-Files/UNSW-NB15-pcap-files/"
		#conf["benign_filename"] = "pcaps-22-1-2015/normal/normal-100000-all-ports.csv"
		conf["benign_filename"] = "pcaps-22-1-2015/normal/TcpTrace/training-no-na.csv"
		conf["malicious_filenames"] = ["pcaps-22-1-2015/attack/TcpTrace/22-1-2015-Backdoors.pcap.csv",
					   "pcaps-22-1-2015/attack/TcpTrace/22-1-2015-Exploits.pcap.csv",
					   "pcaps-22-1-2015/attack/TcpTrace/22-1-2015-Shellcode.pcap.csv",
					   "pcaps-22-1-2015/attack/TcpTrace/22-1-2015-Worms.pcap.csv",
					   "pcaps-17-2-2015/attack/TcpTrace/17-2-2015-Backdoors.pcap.csv",
					   "pcaps-17-2-2015/attack/TcpTrace/17-2-2015-Exploits.pcap.csv",
					   "pcaps-17-2-2015/attack/TcpTrace/17-2-2015-Shellcode.pcap.csv",
					   "pcaps-17-2-2015/attack/TcpTrace/17-2-2015-Worms.pcap.csv"]
		conf["metasploit_filenames"] = "/home/baskoro/exwindows/Datasets/Metasploit/TcpTrace/Pcap/all_filtered_exploit_full.pcap.csv"

		conf["benign_time_filename"] = "normal-full.csv"
		conf["malicious_time_filenames"] = ["1-22-exploits.csv",
					   "1-22-shellcode.csv",
					   "1-22-worms.csv",
					   "1-22-backdoors.csv",
					   "2-17-exploits.csv",
					   "2-17-shellcode.csv",
					   "2-17-worms.csv",
					   "2-17-backdoors.csv",]

		conf["metasploit_time_filenames"] = "metasploit.csv"
		conf["columns_to_remove"] = ['host_a',
					 'host_b',
					 'port_a',
					 'port_b',
					 'SYN/FIN_pkts_sent_a2b',
					 'SYN/FIN_pkts_sent_b2a',
					 'req_1323_ws/ts_a2b',
					 'req_1323_ws/ts_b2a',
					 'adv_wind_scale_a2b',
					 'adv_wind_scale_b2a',
					 'req_sack_a2b',
					 'req_sack_b2a',
					 'Unnamed: 139',
					 'ttl_stream_length_a2b',
					 'ttl_stream_length_b2a',
					 'missed_data_a2b',
					 'missed_data_b2a',
					 'idletime_max_b2a',
					 'first_packet', 
					 'last_packet']
		conf["dst_port_column_name"] = "port_b"
	elif tool == "argus":
		print("not yet supported")
		pass
	elif tool == "bro":
		print("not yet supported")
		pass
	else:
		raise Error("Invalid tool")

	return conf


def read_argus_file(filename):
	df = pd.read_csv(filename)
	return df

def read_bro_file(filename):
	headers = ["ts", "uid", "id_orig_h", "id_orig_p", "id_resp_h", "id_resp_p", 
			   "proto", "service", "duration", "orig_bytes", "resp_bytes", "conn_state", 
			   "local_orig", "local_resp", "missed_bytes", "history", "orig_pkts", "orig_ip_bytes", 
			   "resp_pkts", "resp_ip_bytes", "tunnel_parents"]
	df = pd.read_csv(filename, sep="\t", header=None, comment="#", index_col=False, names=headers)
	return df

def read_tcptrace_file(filename):
	df = pd.read_csv(filename, skiprows=[0,1,2,3,4,5,6,7,9], index_col=False)
	df = df.drop("conn_#", 1)
	return df


def load_data(tool):
	conf = init_conf(tool)

	print("Loading data...")
	if tool == "tcptrace":
		benign_set = read_tcptrace_file(conf["dataset_root_dir"] + conf["benign_filename"])
		malicious_set = read_tcptrace_file(conf["dataset_root_dir"] + conf["malicious_filenames"][0])
		for i in range(1, len(conf["malicious_filenames"])):
			tmp = read_tcptrace_file(conf["dataset_root_dir"] + conf["malicious_filenames"][i])
			malicious_set = malicious_set.append(tmp)
		metasploit_set = read_tcptrace_file(conf["metasploit_filenames"])
	elif tool == "argus":
		pass
	elif tool == "bro":
		pass

	return (conf, benign_set, malicious_set, metasploit_set)


def prepare_data(benign_set, malicious_set, metasploit_set, columns_to_remove):
	print("Preparing data...")
	X = benign_set.append(malicious_set, ignore_index=True, sort=False)
	X = X.drop(columns_to_remove, axis=1)
	X = X.dropna(axis=0)
	X_scaled = preprocessing.scale(X)
	X_meta = metasploit_set
	X_meta = X_meta.drop(columns_to_remove, axis=1)
	X_meta = X_meta.dropna(axis=0)
	X_meta_scaled = preprocessing.scale(X_meta)

	Y = np.zeros(len(benign_set))
	Y = np.append(Y, np.ones(len(X) - len(benign_set)))
	Y_meta = np.ones(len(metasploit_set))

	return X, X_scaled, X_meta, X_meta_scaled, Y, Y_meta


def get_a2b_features(sorted_features, n_features=0):
	a2b_sorted_features = []
	
	for feature, variance in sorted_features:
		if "a2b" in feature and variance > 0:
			a2b_sorted_features.append(feature)

	if n_features == 0:
		n_features = len(a2b_sorted_features)
			
	return a2b_sorted_features[:n_features]


def rank_features(X, Y, benign_set):
	print("Ranking features...")
	var_threshold = VarianceThreshold()
	fit = var_threshold.fit(X[:len(benign_set)].values)
	mask_selected_indices = fit.get_support(indices=True)

	sorted_features = [(x, _) for _,x in sorted(zip(fit.variances_,X.columns), reverse=True)]
	return sorted_features


def split_set(X, Y):
	skf = StratifiedKFold(n_splits=5, shuffle=True)
	indices = skf.split(X, Y)

	indices_list = []

	for train_indices, test_indices in indices:
		indices_list.append((train_indices, test_indices))

	return indices_list


def get_scores(Y_true, Y_pred):
	scores = {}
	#scores["f1"] = metrics.f1_score(Y_true, Y_pred)
	#scores["precision"] = metrics.precision_score(Y_true, Y_pred)
	#scores["recall"] = metrics.recall_score(Y_true, Y_pred)
	#scores["accuracy"] = metrics.accuracy_score(Y_true, Y_pred)
	scores["tn"], scores["fp"], scores["fn"], scores["tp"] = metrics.confusion_matrix(Y_true, Y_pred, labels=[0,1]).ravel()
	return scores
	
def get_avg_score(scores):
	avg_scores = {}
	for key in scores[0]:
		avg_scores[key] = 0
	
	for score in scores:
		for key, value in score.items():
			avg_scores[key] += value

	for key, value in avg_scores.items():
		avg_scores[key] /= float(len(scores))

	return avg_scores


def train_and_predict_one_class(clf, X, Y, X_meta, Y_meta):
	counter = 1
	scores = []
	scores_meta = []
	indices_list = split_set(X, Y)
	
	for train_indices, test_indices in indices_list:
		#print("Fold: {}".format(counter))

		counter += 1
		
		X_train = X[train_indices]
		Y_train = Y[train_indices]
		X_train = X_train[np.where(Y_train==0)[0]]
				
		X_test = X[test_indices]
		Y_test = Y[test_indices]
		
		model = clf.fit(X_train)
		Y_pred = clf.predict(X_test)
		Y_pred = np.where(Y_pred==1, 0, Y_pred)
		Y_pred = np.where(Y_pred==-1, 1, Y_pred)

		Y_pred_meta = clf.predict(X_meta)
		Y_pred_meta = np.where(Y_pred_meta==1, 0, Y_pred_meta)
		Y_pred_meta = np.where(Y_pred_meta==-1, 1, Y_pred_meta)

		scores.append(get_scores(Y_test, Y_pred))
		scores_meta.append(get_scores(Y_meta, Y_pred_meta))

	avg_scores = get_avg_score(scores)
	avg_scores_meta = get_avg_score(scores_meta)
	
	return avg_scores, avg_scores_meta


def predict_on_sorted_features_variance(clf, sorted_features, X, Y, X_meta, Y_meta, a2b_only=False):
	if a2b_only:
		sorted_features = get_a2b_features(sorted_features)
	else:
		sorted_features = [feature for feature, variance in sorted_features]
	
	max_n_features = len(sorted_features)
	dr_fp_all = []
	
	for i in range(1, max_n_features):
		features = sorted_features[:i]
		#print("Analysing " + ",".join(features))
		selectedK_X = X[features]
		selectedK_X_meta = X_meta[features]
		avg_results = train_and_predict_one_class(clf, selectedK_X.values, Y, selectedK_X_meta.values, Y_meta)
		dr_val = avg_results[0]["tp"] / (avg_results[0]["tp"] + avg_results[0]["fn"]) * 100
		fp_val = avg_results[0]["fp"] / (avg_results[0]["fp"] + avg_results[0]["tn"]) * 100
		dr_meta = avg_results[1]["tp"] / (avg_results[1]["tp"] + avg_results[1]["fn"]) * 100
		dr_fp_all.append([dr_val, fp_val, dr_meta])
		
	return np.asarray(dr_fp_all)


def plot_dr_fp(clf, X, Y, X_meta, Y_meta, benign_set, filename):
	sorted_features = rank_features(X, Y, benign_set)

	with open("results/" + filename + "-features.txt", "w") as f_features:
		for feature, variance in sorted_features:
			f_features.write("{},{}\n".format(feature, variance))

	dr_fp_all = predict_on_sorted_features_variance(clf, sorted_features, X, Y, X_meta, Y_meta)
	np.savetxt("results/" + filename + ".csv", dr_fp_all)
	
	fig, axs = plt.subplots(3, 1)
	fig.set_size_inches(20, 20)

	axs[0].plot(dr_fp_all[:,0])
	axs[0].set_xlabel("# Features")
	axs[0].set_ylabel("Detection Rate (%)")
	axs[0].set_ylim([0, 110])
	axs[0].set_xlim([0, 50])
	axs[0].set_title("Detection Rate on UNSW dataset")
	axs[1].plot(dr_fp_all[:,1])
	axs[1].set_xlabel("# Features")
	axs[1].set_ylabel("False Positive Rate (%)")
	axs[1].set_ylim([0, 110])
	axs[1].set_xlim([0, 50])
	axs[1].set_title("False Positive Rate on UNSW dataset")
	axs[2].plot(dr_fp_all[:,2])
	axs[2].set_xlabel("# Features")
	axs[2].set_ylabel("Detection Rate (%)")
	axs[2].set_ylim([0, 110])
	axs[2].set_xlim([0, 50])
	axs[2].set_title("Detection Rate on Metasploit dataset")

	plt.savefig("results/" + filename + ".png")
	print("Finished")


def split_by_dst_port(dataset, conf):
	datasets = {}
	dst_ports = dataset[conf["dst_port_column_name"]].unique().tolist()

	for dst_port in dst_ports:
		if dst_port < 1024:
			datasets[dst_port] = dataset[dataset[conf["dst_port_column_name"]] == dst_port]
		elif 65536 in datasets:
			datasets[65536].append(dataset[dataset[conf["dst_port_column_name"]] == dst_port])
		else:
			datasets[65536] = dataset[dataset[conf["dst_port_column_name"]] == dst_port]

	return datasets


def main(argv):
	try:
		methods = [IsolationForest(), LocalOutlierFactor(novelty=True), svm.OneClassSVM(gamma="auto")]
		filenames = ["if", "lof", "ocsvm"]

		tool = argv[1]
		method_index = int(argv[2])
		if len(argv) > 3:
			model_type = argv[3]
		else:
			model_type = "e"


		conf, benign_set, malicious_set, metasploit_set = load_data(tool)

		if model_type == "e":
			X, X_scaled, X_meta, X_meta_scaled, Y, Y_meta = prepare_data(benign_set, malicious_set, metasploit_set, conf["columns_to_remove"])
			plot_dr_fp(methods[method_index], X, Y, X_meta, Y_meta, benign_set, filenames[method_index])
		else:
			print("Splitting data based on destination ports")
			benign_sets = split_by_dst_port(benign_set, conf)
			malicious_sets = split_by_dst_port(malicious_set, conf)
			metasploit_sets = split_by_dst_port(metasploit_set, conf)
			
			dports = benign_sets.keys()
			print(dports)
			for dport, b_set in benign_sets.items():
				if dport in malicious_sets and dport in metasploit_sets:
					print("Training a model for port " + str(dport))
					ma_set = malicious_sets[dport]
					me_set = metasploit_sets[dport]
					X, X_scaled, X_meta, X_meta_scaled, Y, Y_meta = prepare_data(b_set, ma_set, me_set, conf["columns_to_remove"])
					
					plot_dr_fp(methods[method_index], X, Y, X_meta, Y_meta, b_set, filenames[method_index] + "-a2b0-" + str(dport))
	except IndexError as e:
		print(traceback.print_exc())
		print("Usage: python3 outlier-detection.py tool method_index[0:3] [(e)nsemble|(i)ndividual]")


if __name__ == "__main__":
	main(sys.argv)
