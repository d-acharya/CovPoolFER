##################################################
# SFEW-dataset
##################################################
# Model-4
#python src/classify_sfew.py CLASSIFY data/SFEW_100/Val/ models/model4/20170812-004516/ models/model4/svc_004516_sfewfc2.pkl --batch_size=128 --image_size=100
# Model-2
#python src/classify_sfew.py CLASSIFY data/SFEW_100/Val/ models/model2/20170815-144407/ models/model2/svc_144407_sfewfc2.pkl --batch_size=128 --image_size=100
# Model-baseline
#python src/classify_sfew.py CLASSIFY data/SFEW_100/Val/ models/baseline/20170919-192702/ models/baseline/svc_192702_sfewfc2.pkl --batch_size=128 --image_size=100
# Model-inception-resnet-finetuned # Note that unlike above methods, even for sfew final fc-layer is used to train classifier as there is no penultimate fc layer in inception network.
#python src/classify_sfew_incep.py CLASSIFY data/SFEW_160/Val/ models/incep-tune/20170829-030550/ models/incep-tune/svc_030550_sfew.pkl --batch_size=128 --image_size=160
# Model-inception-resnet-trained
#python src/classify_sfew_incep.py CLASSIFY data/SFEW_160/Val/ models/incep-train/20170830-182116/ models/incep-train/svc_182116_sfew.pkl --batch_size=128 --image_size=160
###################################################
# RAFDB-dataset
###################################################
# Model-4
#python src/classify_rafdb.py CLASSIFY data/RAFDB_100/test/ models/model4/20170812-004516/ models/model4/svc_004516.pkl --batch_size=128 --image_size=100
# Model-2
#python src/classify_rafdb.py CLASSIFY data/RAFDB_100/test/ models/model2/20170815-144407/ models/model2/svc_144407.pkl --batch_size=128 --image_size=100
# Model-baseline
#python src/classify_rafdb.py CLASSIFY data/RAFDB_100/test/ models/baseline/20170919-192702/ models/baseline/svc_192702.pkl --batch_size=128 --image_size=100
# Model-inception-resnet-finetuned
#python src/classify_rafdb.py CLASSIFY data/RAFDB_160/test/ models/incep-tune/20170829-030550/ models/incep-tune/svc_030550.pkl --batch_size=128 --image_size=160
# Model-inception-resnet-trained
#python src/classify_rafdb.py CLASSIFY data/RAFDB_160/test/ models/incep-train/20170830-182116/ models/svc_182116.pkl --batch_size=128 --image_size=160
