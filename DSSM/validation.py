from networks.subgroups import loadModel
from utils import loadText,raiseLogger
import tensorflow as tf
from tensorflow.python import keras
from dataProcess import csvLineCounter,loadTrainData
import numpy as np

batch_size=50
top_ks=[100,200,300,400,500,600,700,800,900,1000]
log=raiseLogger("logs/validation")

# # load dataset
cargo_format=loadText("dataset/validation_cargo_data_format")
cargo_data_path="dataset/validation_data_item_part_20220516.csv"
cargo_data=loadTrainData(cargo_data_path,cargo_format,batch_size=csvLineCounter(cargo_data_path)-1)
cargo_features,_=next(cargo_data)

user_format=loadText("dataset/validation_user_data_format")
# user_data_path="dataset/validation_data_user_part_deal_pre.csv"
# user_data_path="dataset/validation_data_user_part_deal_without_realtime_features.csv"
user_data_path="dataset/validation_data_user_part_deal_20220516.csv"
user_data_counts=csvLineCounter(user_data_path)-1
# batch_size=user_data_counts
user_data=loadTrainData(user_data_path,user_format,batch_size=batch_size)
# user_data=loadTrainData(user_data_path,user_format,batch_size=user_data_counts)
user_features,_=next(user_data)

# # load model

# cargo_tower=loadModel("weights/weights_v2/item_tower_20220330084953")
# user_tower=loadModel("weights/weights_v2/user_tower_20220330084953")
# cargo_tower=loadModel("past_codes/saves/v2/item_tower_20220309140146")
# user_tower=loadModel("past_codes/saves/v2/user_tower_20220309140146")
# cargo_tower=loadModel("past_codes/saves/v3/item_tower_20220309142423")
# user_tower=loadModel("past_codes/saves/v3/user_tower_20220309142423")
# cargo_tower=loadModel("weights/weights_v4/item_tower_20220309140530")
# user_tower=loadModel("weights/weights_v4/user_tower_20220309140530")
# cargo_tower=loadModel("weights/weights_v5/item_tower_20220328122629")
# user_tower=loadModel("weights/weights_v5/user_tower_20220328122629")
# cargo_tower=loadModel("weights/weights_v6/item_tower_20220331105856")
# user_tower=loadModel("weights/weights_v6/user_tower_20220331105856")
# cargo_tower=loadModel("weights/weights_v7/item_tower_20220328132301")
# user_tower=loadModel("weights/weights_v7/user_tower_20220328132301")
# cargo_tower=loadModel("weights/weights_v8/item_tower_20220331061500")
# user_tower=loadModel("weights/weights_v8/user_tower_20220331061500")
# cargo_tower=loadModel("weights/weights_v9/item_tower_20220331104535")
# user_tower=loadModel("weights/weights_v9/user_tower_20220331104535")
# cargo_tower=loadModel("weights/weights_v10/item_tower_20220331075744")
# user_tower=loadModel("weights/weights_v10/user_tower_20220331075744")
cargo_tower=loadModel("past_codes/saves/v11/item_tower_20220406124722")
user_tower=loadModel("past_codes/saves/v11//user_tower_20220406124722")
# cargo_tower=loadModel("weights/weights_v11/item_tower_20220420104105")
# user_tower=loadModel("weights/weights_v11/user_tower_20220420104105")
# cargo_tower=loadModel("weights/weights_v12/item_tower_20220406152455")
# user_tower=loadModel("weights/weights_v12/user_tower_20220406152455")
# cargo_tower=loadModel("weights/weights_v13/item_tower_20220509124711")
# user_tower=loadModel("weights/weights_v13/user_tower_20220509124711")
# cargo_tower=loadModel("weights/weights_v14/item_tower_20220508171406")
# # user_tower=loadModel("weights/weights_v14/user_tower_20220508171406")
# cargo_tower=loadModel("weights/weights_v15/item_tower_20220512084231")
# user_tower=loadModel("weights/weights_v15/user_tower_20220512084231")



#train
cargo_embedding=cargo_tower(cargo_features)
# cargo_embedding=tf.nn.l2_normalize(cargo_embedding,axis=-1) #仅测试用
cargo_embedding=tf.transpose(cargo_embedding)
epochs=user_data_counts//batch_size
log.info(f"data will loop {epochs} epochs")
ttl_cover_num={}
ttl_case_num={}
ttl_road_1_cross={}
ttl_road_2_cross={}
ttl_road_n3_cross={}
for top_k in top_ks:
    ttl_cover_num[top_k]=0
    ttl_case_num[top_k]=0
    ttl_road_1_cross[top_k]=0
    ttl_road_2_cross[top_k]=0
    ttl_road_n3_cross[top_k]=0
print("ready")
for i in range(epochs):
    user_embedding=user_tower(user_features)
    # user_embedding=tf.nn.l2_normalize(user_embedding,axis=-1) #仅测试用
    cover_num={}
    case_num={}
    road_1_cross={}
    road_2_cross={}
    road_n3_cross={}
    for top_k in top_ks:
        cover_num[top_k]=0
        case_num[top_k]=0
        road_1_cross[top_k]=0
        road_2_cross[top_k]=0
        road_n3_cross[top_k]=0
    yp=tf.add(tf.matmul(user_embedding,cargo_embedding),1.0)/2
    user_infos=tf.concat([user_features["user_id"],user_features["session_id"],
        user_features["searchtimestamp"],user_features["cargo_id"],
        user_features["road1_isrecall"],user_features["road2_isrecall"],
        user_features["roadn3_isrecall"]],axis=1)
    user_timestamp=tf.strings.to_number(user_features["searchtimestamp"],out_type=tf.int32)
    cargo_info=tf.concat([cargo_features["cargo_id"]],axis=1)
    cargo_start_timestamp=tf.strings.to_number(cargo_features["create_timestamp"],out_type=tf.int32)
    cargo_end_timestamp=tf.strings.to_number(cargo_features["cargo_end_timestamp"],out_type=tf.int32)
    user_is_later_then_cargo_start=tf.greater(
        (
            tf.matmul(user_timestamp,tf.ones([1,cargo_start_timestamp.shape[0]],dtype=tf.int32))-
            tf.matmul(tf.ones([user_timestamp.shape[0],1],dtype=tf.int32),tf.transpose(cargo_start_timestamp))
        ),
        tf.zeros([user_timestamp.shape[0],cargo_start_timestamp.shape[0]],dtype=tf.int32)
        )

    user_is_earlier_then_cargo_end=tf.greater(
        tf.zeros([user_timestamp.shape[0],cargo_end_timestamp.shape[0]],dtype=tf.int32),
        (
            tf.matmul(user_timestamp,tf.ones([1,cargo_end_timestamp.shape[0]],dtype=tf.int32))-
            tf.matmul(tf.ones([user_timestamp.shape[0],1],dtype=tf.int32),tf.transpose(cargo_end_timestamp))
        )
        )
    # cargo_is_in_12_hour=tf.greater(
    #     tf.matmul(tf.ones([user_timestamp.shape[0],1],dtype=tf.int32),tf.transpose(cargo_start_timestamp)),
    #     tf.matmul(user_timestamp,tf.ones([1,cargo_start_timestamp.shape[0]],dtype=tf.int32))-43200
    # )
    # user_is_later_then_cargo_start=tf.logical_and(user_is_later_then_cargo_start,cargo_is_in_12_hour)
    user_valid_cargo=tf.logical_and(user_is_later_then_cargo_start,user_is_earlier_then_cargo_end)
    valid_scores=tf.where(user_valid_cargo,yp,-1.0)
    sorted_cargo=tf.argsort(valid_scores,direction="DESCENDING",axis=1)
    for top_k in top_ks:
        top_k_cargo=sorted_cargo[:,:top_k]
        top_k_cargo_id=tf.gather(tf.squeeze(cargo_info),top_k_cargo)
        # road_1_overlap_num=
        for user_info,tops in zip(user_infos,top_k_cargo_id):
            user_picked_cargo_id=user_info[-4]
            if user_picked_cargo_id.numpy() in tops.numpy():
                road_1_cross[top_k]+=int(user_info[-3])
                road_2_cross[top_k]+=int(user_info[-2])
                road_n3_cross[top_k]+=int(user_info[-1])
                cover_num[top_k]+=1
            case_num[top_k]+=1
        ttl_road_1_cross[top_k]+=road_1_cross[top_k]
        ttl_road_2_cross[top_k]+=road_2_cross[top_k]
        ttl_road_n3_cross[top_k]=road_n3_cross[top_k]
        ttl_cover_num[top_k]+=cover_num[top_k]
        ttl_case_num[top_k]+=case_num[top_k]

        log.info(f"top_k {top_k}:cover_num {cover_num[top_k]},case_num {case_num[top_k]}")
        log.info(f"epoch {i}: top_k {top_k}:epoch_click_recall_rate={cover_num[top_k]/case_num[top_k]} average_click_recall_rate={ttl_cover_num[top_k]/ttl_case_num[top_k]}")
        log.info(f"epoch {i}: top_k {top_k}:road 1 cross rate:{ttl_road_1_cross[top_k]/ttl_cover_num[top_k]},\
                 road 2 cross rate:{ttl_road_2_cross[top_k]/ttl_cover_num[top_k]},\
                 road n3 cross rate:{ttl_road_n3_cross[top_k]/ttl_cover_num[top_k]}")
    user_features,_=next(user_data)