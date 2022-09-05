from recbole.quick_start import run_recbole

run_recbole(model="MLP4Rec",dataset="ml-100k",config_dict={"epochs":100,"neg_sampling":None,"train_batch_size":256,"eval_batch_size":512,"hidden_dropout_prob":0.5,"learning_rate":0.0001,"n_layers":4,"show_progress":True,"hidden_size":128,"seed":2022})
