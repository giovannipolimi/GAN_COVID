from gan_covid_packages import *

from gan_covid_main import location_dummy
from gan_covid_main import country_dummy
from gan_covid_main import gender_dummy
from gan_covid_main import vis_wuhan_dummy
from gan_covid_main import from_wuhan_dummy
from gan_covid_main import symptom1_dummy
from gan_covid_main import symptom2_dummy
from gan_covid_main import symptom3_dummy
from gan_covid_main import symptom4_dummy
from gan_covid_main import symptom5_dummy
from gan_covid_main import symptom6_dummy
from gan_covid_main import numerical_df_rescaled

def define_generator (catsh1,catsh2,catsh3,catsh4,catsh5,catsh6,catsh7,catsh8,catsh9,catsh10,catsh11,numerical):    
  #Inputting noise  from latent space
    noise = Input(shape = (70,))    
    hidden_1 = Dense(8, kernel_initializer = "he_uniform")(noise)    
    hidden_1 = LeakyReLU(0.2)(hidden_1)    
    hidden_1 = BatchNormalization(momentum = 0.8)(hidden_1)    
    hidden_2 = Dense(16, kernel_initializer = "he_uniform")(hidden_1)    
    hidden_2 = LeakyReLU(0.2)(hidden_2)    
    hidden_2 = BatchNormalization(momentum = 0.8)(hidden_2)    

    #Branch 1 for generating location data

    branch_1 = Dense(32, kernel_initializer = "he_uniform")(hidden_2)    
    branch_1 = LeakyReLU(0.2)(branch_1)    
    branch_1 = BatchNormalization(momentum = 0.8)(branch_1)    
    branch_1 = Dense(64, kernel_initializer = "he_uniform")(branch_1)    
    branch_1 = LeakyReLU(0.2)(branch_1)    
    branch_1 = BatchNormalization(momentum=0.8)(branch_1)    
 
    #Output Layer1
    branch_1_output = Dense(catsh1, activation = "softmax")(branch_1)    

    #Likewise, for all remaining 10 categories branches will be defined    
    #Branch 12 for generating numerical data 
    branch_12 = Dense(64, kernel_initializer = "he_uniform")(hidden_2)    
    #branch_12 = LeakyReLU(0.2)(branch_3)   
    branch_12 = LeakyReLU(0.2)(branch_12)    
    branch_12 = BatchNormalization(momentum=0.8)(branch_12)    
    branch_12 = Dense(128, kernel_initializer = "he_uniform")(branch_12)    
    branch_12 = LeakyReLU(0.2)(branch_12)    
    branch_12 = BatchNormalization(momentum=0.8)(branch_12)    
    
    #Output Layer12 
    branch_12_output = Dense(numerical, activation = "sigmoid")(branch_12)    

    #Combined output 
    combined_output = concatenate([branch_1_output, branch_2_output, branch_3_output,branch_4_output,branch_5_output,branch_6_output,branch_7_output,branch_8_output,branch_9_output,branch_10_output,branch_11_output,branch_12_output])    

    #Return model 

    return Model(inputs = noise, outputs = combined_output)    

    
generator = define_generator(location_dummy.shape[1],country_dummy.shape[1],gender_dummy.shape[1],vis_wuhan_dummy.shape[1],from_wuhan_dummy.shape[1],symptom1_dummy.shape[1],symptom2_dummy.shape[1],symptom3_dummy.shape[1],symptom4_dummy.shape[1],symptom5_dummy.shape[1],symptom6_dummy.shape[1],numerical_df_rescaled.shape[1])  
generator.summary()

