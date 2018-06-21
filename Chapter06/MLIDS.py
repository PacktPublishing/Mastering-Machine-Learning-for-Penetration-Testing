import pandas as pd
import yellowbrick as yb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split



Columns = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]


Data = pd.read_csv("KDDTrain+.csv", header=None, names = Columns) 

Data.protocol_type = preprocessing.LabelEncoder().fit_transform(Data["protocol_type"])
Data.service = preprocessing.LabelEncoder().fit_transform(Data["service"])
Data.flag = preprocessing.LabelEncoder().fit_transform(Data["flag"])
Data.label = preprocessing.LabelEncoder().fit_transform(Data["label"])

X = Data[Columns].as_matrix()
y = Data.label.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
Score = clf.score(X_test,y_test)
print(Score*100)
