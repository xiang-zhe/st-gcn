Since there is a global pooling layer at the end of ST-GCN. You can feed sequences with arbitrary length. We do not observe considerable performance degradation or improvement when the input shape changes. We make length=300 because the maximal length of videos in Kinetics and NTU-RGB-D dataset is 300. windows_size

st-gcn/processor/demo.py:
    output, feature = self.model.extract_feature(data) 
    output = output[0] 
    ...
    label = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0) 
    
st-gcn/processor/demo.py:
    video_info = utils.openpose.json_pack(output_snippets_dir, video_name, width, height)
    
kinetics-gendata.py
        num_person_in=5,  #observe the first 5 persons 
        num_person_out=2,  #then choose 2 persons with the highest score    

feeder_kinetics.py
    # fill data_numpy
                data_numpy[0, frame_index, :, m] = pose[0::2]
                data_numpy[1, frame_index, :, m] = pose[1::2]
                data_numpy[2, frame_index, :, m] = score
    这里并没有做300帧的补齐，而是padding 0
            # data augmentation
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)    
            
tool.py
    random_chooose:这里对大于尺寸的进行截取，小于尺寸的如果pad=true就补齐0
    
    auto_pading:这里对小于尺寸的使用补0 pading，对大于尺寸的未处理
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
 

ValueError: mmap length is greater than file size:
数据问题，重新生成数据集

修改类别数：
    feeder—kinetics.py line71-74
        '''
        self.label = np.array(
            [label_info[id]['label_index'] for id in sample_id])
        has_skeleton = np.array(
            [label_info[id]['has_skeleton'] for id in sample_id])
        '''
        self.label = np.array(
            [label_info[id]['label_index'] for id in sample_id if label_info[id]['label_index'] <40])  ##  label
        has_skeleton = np.array(
            [label_info[id]['has_skeleton'] for id in sample_id if label_info[id]['label_index'] <40])
        self.sample_name = ['.'.join([id, 'json']) for id in sample_id if label_info[id]['label_index'] <40] ##  skeleton filename
    or modified orginial json -- kinetics-train，kinetics-val，kinetics_train_label.json，kinetics_val_label.json

/pytorch/aten/src/THCUNN/ClassNLLCriterion.cu:105: void cunn_ClassNLLCriterion_updateOutput_kernel(Dtype *, Dtype *, Dtype *, long *, Dtype *, int, int, int, int, long) [with Dtype = float, Acctype = float]: block: [0,0,0], thread: [29,0,0] Assertion `t >= 0 && t < n_classes` failed.
THCudaCheck FAIL file=/pytorch/aten/src/THC/generic/THCStorage.c line=36 error=59 : device-side assert triggered
RuntimeError: cuda runtime error (59) : device-side assert triggered when running transfer_learning_tutorial
    原因是计算损失的时候label不是从0开始的或连续的，pytorch必须是0-n的label才不会抛错
    kinetics-gendata.py的line57 for i, s in enumerate(sample_name):循环中制定标签

sample_weight: 
    recogntion.py中添加loss中的weight，具体参见demo
        #class_weight = torch.from_numpy(np.array([]))
        class_weight = torch.ones(40)
        class_weight[-1] = 15
        self.loss = nn.CrossEntropyLoss(weight=class-weight)


python tools/kinetics_gendata.py --data_path /media/xiang/581ff64d-4b67-074d-af77-4bf4ee44b3f7/dataset/st-gcn/kinetics-skeleton

demo:
python main.py demo --openpose ../openpose/build

train:
python main.py recognition -c config/st_gcn/kinetics-skeleton/train.yaml --device 0 --batch_size 32 --test_batch_size 32 --weights /home/xiang/git/st-gcn/work_dir/recognition/kinetics_skeleton/ST_GCN/epoch90_model.pt



