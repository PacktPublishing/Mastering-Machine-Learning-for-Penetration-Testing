

from cleverhans.utils_tf import model_train , model_eval , batch_eval
from cleverhans.attacks_tf import jacobian_graph
from cleverhans.utils import other_classes
from cleverhans.utils_tf import model_train , model_eval , batch_eval
from cleverhans.attacks_tf import jacobian_graph
from cleverhans.utils import other_classes
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , roc_curve , auc , f1_score
from sklearn.preprocessing import LabelEncoder , MinMaxScaler
from tensorflow.python.platform import flags

names = ['duration', 'protocol', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land','wrong_fragment','urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted','num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds','is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate','rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate','dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate','dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate', 'dst_host_srv_rerror_rate','attack_type', 'other']

df = pd.read_csv('KDDTrain+.txt', names=names , header=None)
dft = pd.read_csv('KDDTest+.txt', names=names , header=None)


FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size ', 128, 'Size of training batches ')
flags.DEFINE_float('learning_rate ', 0.1, 'Learning rate for training ')
flags.DEFINE_integer('nb_classes ', 5, 'Number of classification classes ')
flags.DEFINE_integer('source_samples ', 10, 'Nb of test set examples to attack ')
full = pd.concat([df,dft])
assert full.shape[0] == df.shape[0] + dft.shape[0]
print("Initial test and training data shapes:", df.shape , dft.shape)

full.loc[full.label == 'neptune ', 'label'] = 'dos'
full.loc[full.label == 'back', 'label'] = 'dos'
full.loc[full.label == 'land', 'label'] = 'dos'
full.loc[full.label == 'pod', 'label'] = 'dos'
full.loc[full.label == 'smurf', 'label'] = 'dos'
full.loc[full.label == 'teardrop', 'label'] = 'dos'
full.loc[full.label == 'mailbomb', 'label'] = 'dos'
full.loc[full.label == 'processtable', 'label'] = 'dos'
full.loc[full.label == 'udpstorm', 'label'] = 'dos'
full.loc[full.label == 'apache2', 'label'] = 'dos'
full.loc[full.label == 'worm', 'label'] = 'dos'

full.loc[full.label == 'buffer_overflow', 'label'] = 'u2r'
full.loc[full.label == 'loadmodule', 'label'] = 'u2r'
full.loc[full.label == 'perl', 'label'] = 'u2r'
full.loc[full.label == 'rootkit', 'label'] = 'u2r'
full.loc[full.label == 'sqlattack', 'label'] = 'u2r'
full.loc[full.label == 'xterm', 'label'] = 'u2r'
full.loc[full.label == 'ps', 'label'] = 'u2r'

full.loc[full.label == 'ftp_write', 'label'] = 'r2l'
full.loc[full.label == 'guess_passwd', 'label'] = 'r2l'
full.loc[full.label == 'imap', 'label'] = 'r2l'
full.loc[full.label == 'multihop', 'label'] = 'r2l'
full.loc[full.label == 'phf', 'label'] = 'r2l'
full.loc[full.label == 'spy', 'label'] = 'r2l'
full.loc[full.label == 'warezclient', 'label'] = 'r2l'
full.loc[full.label == 'warezmaster', 'label'] = 'r2l'
full.loc[full.label == 'xlock', 'label'] = 'r2l'
full.loc[full.label == 'xsnoop', 'label'] = 'r2l'
full.loc[full.label == 'snmpgetattack', 'label'] = 'r2l'
full.loc[full.label == 'httptunnel', 'label'] = 'r2l'
full.loc[full.label == 'snmpguess', 'label'] = 'r2l'
full.loc[full.label == 'sendmail', 'label'] = 'r2l'
full.loc[full.label == 'named', 'label'] = 'r2l'


full.loc[full.label == 'satan', 'label'] = 'probe'
full.loc[full.label == 'ipsweep', 'label'] = 'probe'
full.loc[full.label == 'nmap', 'label'] = 'probe '
full.loc[full.label == 'portsweep', 'label'] = 'probe '
full.loc[full.label == 'saint', 'label'] = 'probe'
full.loc[full.label == 'mscan', 'label'] = 'probe'

full = full.drop(['other', 'attack_type'], axis =1)
print("Unique labels", full.label.unique())

full2 = pd.get_dummies(full , drop_first=False)
features = list(full2.columns[:-5])


y_train = np.array(full2[0:df.shape[0]][[ 'label_normal', 'label_dos', 'label_probe', 'label_r2l', 'label_u2r']])
X_train = full2[0:df.shape[0]][features]
y_test = np.array(full2[df.shape[0]:][['label_normal', 'label_dos', 'label_probe', 'label_r2l', 'label_u2r']])
X_test = full2[df.shape[0]:][features]


scaler = MinMaxScaler().fit(X_train)
X_train_scaled = np.array(scaler.transform(X_train))
X_test_scaled = np.array(scaler.transform(X_test))
labels = full.label.unique()
le = LabelEncoder()
le.fit(labels)

y_full = le.transform(full.label)
y_train_l = y_full[0:df.shape[0]]
y_test_l = y_full[df.shape[0]:]

print("Training dataset shape", X_train_scaled.shape , y_train.shape)
print("Test dataset shape", X_test_scaled.shape , y_test.shape)
print("Label encoder y shape", y_train_l.shape , y_test_l.shape)


def mlp_model():

	model = Sequential()
	model.add(Dense(256,activation='relu', input_shape =( X_train_scaled.shape[1],)))
	model.add(Dropout(0.4))
	model.add(Dense(256,  activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(5 , activation='softmax'))
	model.compile(optimizer='adam',metrics =['accuracy'])
	model.summary()
	return  model

def  evaluate():
	eval_params = {'batch_size ': FLAGS.batch_size}
	accuracy = model_eval(sess , x, y, predictions , X_test_scaled , y_test , args= eval_params)
	print('Test  accuracy  on  legitimate  test  examples: ' + str(accuracy))


x = tf.placeholder(tf.float32 , shape=(None ,X_train_scaled.shape[1]))
y = tf.placeholder(tf.float32 , shape=(None ,5))







