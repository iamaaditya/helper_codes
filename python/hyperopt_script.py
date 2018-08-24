from subprocess import call
import random

def do_process():
    all_options = {}
    
    all_options['-learning_rate'             ]=[ 3e-4, 4e-4, 3.5e-4, 3.2e-4, 3e-3, 5e-5]
    all_options['-input_encoding_size'       ]=[ 400, 600, 800, 800, 900, 1000, 1100, 1200, 1400, 1200]
    all_options['-rnn_size'                  ]=[ 512, 256, 512, 1024]
    all_options['-rnn_layer'                 ]=[ 1,2,2,3]
    all_options['-common_embedding_size'     ]=[ 2048, 1024, 1536, 2048, 512]
    all_options['-num_output'                ]=[ 1000, 980, 1000, 990, 1000]
    all_options['-img_norm'                  ]=[ 0, 1, 0]  
    all_options['-non_linearity'             ]=[ 'tanh', 'relu']
    all_options['-initialization'            ]=[ 'uniform', 'heuristic' , 'xavier'  ,  'xavier_caffe', 'kaiming' ]
    all_options['-uniform_init_limit'        ]=[ -6, -2, -1, 0, 1, 2]
    all_options['-extra_layer'               ]=[ 0,1]      
    all_options['-bias_vqa'                  ]=[ 0,1]     
    all_options['-bias_highway'              ]=[ 0,1]
    all_options['-num_highway_embedding'     ]=[ 1, 2,3, 5, 8]   
    all_options['-num_highway_vqa'           ]=[ 1,2,3,4, 6, 10]  
    all_options['-highway_bias_embedding'    ]=[ -6, -2, -1, 0, 1, 2]
    all_options['-highway_bias_vqa'          ]=[ -6, -2, -1, 0, 1, 2]
    all_options['-decay_factor'              ]=[ 0.99999 + 0.0000013*i for i in xrange(1,8)]
    all_options['-dropout'                   ]=[ 0.2*i for i in xrange(5)] 
    all_options['-dropout_eval'              ]=[ 0.2*i for i in xrange(5)] 
    all_options['-kick'                      ]=[ i*8000 for i in xrange(6)]
    all_options['-change'                    ]=[ i*8000 for i in xrange(6)]
    all_options['-kick_value'                ]=[ 0.2*i for i in xrange(6) ] 
    
    # print(all_options.values())
    for k, v in all_options.iteritems():
        print k,v
        
    i = 0
    while True:
        i = i + 1
        s = []
        s.append('th')
        s.append('train_hyper.lua')
        
        for k,v in all_options.iteritems():
            s.append(k)
            s.append(str(random.choice(v)))

        print s
        call(s)
        # call(['th', 'train_hyper.lua'

        # if i > 30: break

do_process()          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
