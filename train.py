from model import Model_Reactionary_Detection

def train(labId = "reactionary_detection",model_name = 'distilbert-base-uncased', train_data_dir = "./datasets/val.csv", val_size = 0.1,
                learning_rate = 1e-5, epochs = 3, batch_size= 64):
    
    model = Model_Reactionary_Detection(labId=labId, model_name=model_name, train_data_dir=train_data_dir,val_size=val_size)
    train_output = model.train(learning_rate=learning_rate,EPOCHS=epochs, BS=batch_size)
    for res_per_epoch in train_output:
	    yield res_per_epoch
    
if __name__ == '__main__':
    respones = train()
    for res in respones:
        print(res)
