import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import tensorflow as tf


def LS_1(w,M,r_1,r_2):
    
    LS_1_value_one = tf.reduce_sum(tf.sign(np.dot(w,M.T)))
    LS_1_value_two = tf.reduce_sum(tf.sign(np.dot(-w,M.T)))
    LS_1_value = max(LS_1_value_one,LS_1_value_two)/(r_1*r_2)
    
    return LS_1_value

def LS_2(w,M,r_1,r_2):
    
    L2_up = np.abs(np.sum(np.dot(w,M.T)))
    L2_down = np.sum(np.abs(np.dot(w,M.T)))
    LS_2_value = L2_up/L2_down
    
    return LS_2_value
    
def J_w(w,M,r_1,r_2):
    
    J_w_up = np.square(np.sum(np.dot(w,M.T)))
    J_w_down = np.sum(np.square(np.dot(w,M.T)))
    J_w_value = J_w_up/J_w_down
    
    return J_w_value

def LDA(w,new_data_A,new_data_B,r_1,r_2):
    
    w = w.T
    
    A_class = new_data_A.numpy()
    B_class = new_data_B.numpy()
    A_class = A_class.T
    B_class = B_class.T

    nu_a = np.mean(A_class,axis=1).reshape(-1,1)
    nu_b = np.mean(B_class,axis=1).reshape(-1,1)

    A_c = A_class-np.repeat(nu_a,A_class.shape[1],axis=1)
    B_c = B_class-np.repeat(nu_b,B_class.shape[1],axis=1)

    S_w = np.dot(A_c,A_c.T)+np.dot(B_c,B_c.T)
    
    S_b = np.dot(nu_a-nu_b,(nu_a-nu_b).T)


    LDA_value_up = np.dot(np.dot(w.T,S_b),w)
    LDA_value_down = np.dot(np.dot(w.T,S_w),w)
    LDA_value = LDA_value_up/LDA_value_down
    
    return LDA_value

def W(original_X,original_Y):
    
    all_class = np.unique(original_Y)
    num_class = len(all_class)
    
    LS_1_value_sequnce = np.array(np.zeros_like(all_class),dtype = 'float32')
    LS_2_value_sequnce = np.array(np.zeros_like(all_class),dtype = 'float32')
    J_w_value_sequnce = np.array(np.zeros_like(all_class),dtype = 'float32')
    LDA_value_sequnce = np.array(np.zeros_like(all_class),dtype = 'float32')
    ratio_sequence = np.array(np.zeros_like(all_class),dtype = 'float32')

    
    for num_class_i in range(num_class):
    
        class_i = all_class[num_class_i]
        
        original_X = tf.constant(original_X)
        original_Y = tf.constant(original_Y)

        data_A = tf.gather(original_X, axis=0, indices=tf.where(original_Y==class_i)[:,0])
        data_B = tf.gather(original_X, axis=0, indices=tf.where(original_Y!=class_i)[:,0])

        r_1 = len(data_A)
        r_2 = len(data_B)

        new_data_A = tf.reshape(data_A,(data_A.shape[0],-1))
        new_data_B = tf.reshape(data_B,(data_B.shape[0],-1))

#         if new_data_A.shape[1]>=1000 or new_data_B.shape[1]>=1000:
#             new_data_A = tf.reduce_mean(data_A,axis=-1)
#             new_data_B = tf.reduce_mean(data_B,axis=-1)
#             new_data_A = tf.reshape(new_data_A,(new_data_A.shape[0],-1))
#             new_data_B = tf.reshape(new_data_B,(new_data_B.shape[0],-1))

        M = np.zeros((r_1*r_2,new_data_A.shape[1]),dtype='float32')

        index_base = np.arange(r_2)

        for i in range(r_1):
            M[index_base+i*r_2,:]=new_data_A[i]-new_data_B  



        m =np.sum(M,axis=0).reshape(1,-1)
        w = m/np.linalg.norm(m)

        LS_1_value_sequnce[num_class_i] = LS_1(w,M,r_1,r_2)
        LS_2_value_sequnce[num_class_i] = LS_2(w,M,r_1,r_2)
        J_w_value_sequnce[num_class_i] = J_w(w,M,r_1,r_2)
        LDA_value_sequnce[num_class_i] = LDA(w,new_data_A,new_data_B,r_1,r_2)
        ratio_sequence[num_class_i] = len(tf.where(original_Y==class_i)[:,0])/len(original_X)
     
    
    LS_1_value = np.sum(LS_1_value_sequnce*ratio_sequence)
    LS_2_value = np.sum(LS_2_value_sequnce*ratio_sequence)
    J_w_value = np.sum(J_w_value_sequnce*ratio_sequence)
    LDA_value = np.sum(LDA_value_sequnce*ratio_sequence)
    
    return LS_1_value,LS_2_value,J_w_value,LDA_value



def get_layer_name(model):
    
    layer_name_list = []
    number_squence = [0,'1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th','11th','12th','13th','14th','15th','16th','17th','18th','19th','20th','21th','22th','23th','24th','25th']

    name_conts = np.ones((9,),dtype='int')
    
    for layer in model.layers: 
        if 'conv2d' in layer.name: 
            name_conts_layer = name_conts[0]
            layer_name_list.append(number_squence[name_conts_layer]+'_'+'conv2d')
            name_conts[0]+=1
        if 'max_pooling' in layer.name: 
            name_conts_layer = name_conts[1]
            layer_name_list.append(number_squence[name_conts_layer]+'_'+'max_pooling')
            name_conts[1]+=1
        if 'flatten' in layer.name: 
            name_conts_layer = name_conts[2]
            layer_name_list.append(number_squence[name_conts_layer]+'_'+'flatten')    
            name_conts[2]+=1
        if 'dense' in layer.name:
            name_conts_layer = name_conts[3]
            layer_name_list.append(number_squence[name_conts_layer]+'_'+'dense')  
            name_conts[3]+=1
        if 'batch_normalization' in layer.name: 
            name_conts_layer = name_conts[4]
            layer_name_list.append(number_squence[name_conts_layer]+'_'+'batch_normalization')     
            name_conts[4]+=1
        if 'dropout' in layer.name: 
            name_conts_layer = name_conts[5]
            layer_name_list.append(number_squence[name_conts_layer]+'_'+'dropout')  
            name_conts[5]+=1
        if 'add' in layer.name: 
            name_conts_layer = name_conts[6]
            layer_name_list.append(number_squence[name_conts_layer]+'_'+'add')  
            name_conts[6]+=1
        if 'activation' in layer.name: 
            name_conts_layer = name_conts[7]
            layer_name_list.append(number_squence[name_conts_layer]+'_'+'activation')  
            name_conts[7]+=1            
        if 'average_pooling2d' in layer.name: 
            name_conts_layer = name_conts[8]
            layer_name_list.append(number_squence[name_conts_layer]+'_'+'average_pooling2d')  
            name_conts[8]+=1       
            
    return layer_name_list





def plot_Separability_figure(layer_name_list,x_plot,LS_1_squence,LS_2_squence,J_w_squence,LDA_squence,LS_1_squence_base,LS_2_squence_base,J_w_squence_base,LDA_squence_base):
    
    
    #figure = plt.figure(1,figsize=(8,8))
    figure = plt.figure(figsize=(16,10))
    #layer_name_list = get_layer_name(model)[reserved_layers:]
    
    ax1 = plt.subplot(2,2,1)

    for i in range(len(LS_1_squence)):
        if 'flatten' not in layer_name_list[i]:
            plt.plot(x_plot,LS_1_squence[i,:],label=layer_name_list[i])
    plt.plot(x_plot,LS_1_squence_base[0,:],'-k',label='base_line')
    plt.title('$\mathrm{LS}_0$')
    #plt.legend(bbox_to_anchor=(1,0),loc=3)

    ax2 = plt.subplot(2,2,2)

    for i in range(len(LS_2_squence)):
        if 'flatten' not in layer_name_list[i]:
            plt.plot(x_plot,LS_2_squence[i,:],label=layer_name_list[i])
    plt.plot(x_plot,LS_2_squence_base[0,:],'-k',label='base_line')
    plt.title('$\mathrm{LS}_1$')
    plt.legend(bbox_to_anchor=(1,0),loc=3)

    ax3 = plt.subplot(2,2,3)

    for i in range(len(J_w_squence)):
        if 'flatten' not in layer_name_list[i]:
            plt.plot(x_plot,J_w_squence[i,:],label=layer_name_list[i])
    plt.plot(x_plot,J_w_squence_base[0,:],'-k',label='base_line')
    plt.title('$\mathrm{LS}_2$')
    #plt.legend(bbox_to_anchor=(1,0),loc=3)

    ax4 = plt.subplot(2,2,4)

    for i in range(len(LDA_squence)):
        if 'flatten' not in layer_name_list[i]:
            plt.plot(x_plot,LDA_squence[i,:],label=layer_name_list[i])
    plt.plot(x_plot,LDA_squence_base[0,:],'-k',label='base_line')
    plt.title('$\mathrm{LDA}$')
    plt.legend(bbox_to_anchor=(1,0),loc=3)
    
    plt.tight_layout()
    
    
    
    return figure
    
    
def plot_net_figure(layer_name_list,x_plot,train_loss_squence,train_accuracy_squence,test_loss_squence,test_accuracy_squence):
    
    
    #figure = plt.figure(1,figsize=(8,8))
    figure = plt.figure(figsize=(16,10))
    #layer_name_list = get_layer_name(model)[reserved_layers:]
    
    ax1 = plt.subplot(2,2,1)
    plt.plot(x_plot,train_loss_squence)
    plt.title('Train Loss')

    ax2 = plt.subplot(2,2,2)
    plt.plot(x_plot,train_accuracy_squence)
    plt.title('Train Accuracy')

    ax3 = plt.subplot(2,2,3)
    plt.plot(x_plot,test_loss_squence)
    plt.title('Test Loss')


    ax4 = plt.subplot(2,2,4)
    plt.plot(x_plot,test_accuracy_squence)
    plt.title('Test Accuracy')

    
    plt.tight_layout()
    
    return figure


