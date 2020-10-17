from gan_covid_packages import *
from gan_covid_source import df

print(df.head)
# Analyse only deaths
df_minority_data = df.loc[df['death'] == 1]

#Subsetting input features without target variable
df_minority_data_withouttv = df_minority_data.loc[:,
                                                  df_minority_data.columns != 'death']

numerical_df = df_minority_data_withouttv.select_dtypes("number")
categorical_df = df_minority_data_withouttv.select_dtypes("object")
scaling = MinMaxScaler()
numerical_df_rescaled = scaling.fit_transform(numerical_df)
get_dummy_df = pd.get_dummies(categorical_df)

#Seperating Each Category
location_dummy_col = [col for col in get_dummy_df.columns if 'location' in col]
location_dummy = get_dummy_df[location_dummy_col]
country_dummy_col = [col for col in get_dummy_df.columns if 'country' in col]
country_dummy = get_dummy_df[country_dummy_col]
gender_dummy_col = [col for col in get_dummy_df.columns if 'gender' in col]
gender_dummy = get_dummy_df[gender_dummy_col]
vis_wuhan_dummy_col = [col for col in get_dummy_df.columns if 'vis_wuhan' in col]
vis_wuhan_dummy = get_dummy_df[vis_wuhan_dummy_col]
from_wuhan_dummy_col = [col for col in get_dummy_df.columns if 'from_wuhan' in col]
from_wuhan_dummy = get_dummy_df[from_wuhan_dummy_col]
symptom1_dummy_col = [col for col in get_dummy_df.columns if 'symptom1' in col]
symptom1_dummy = get_dummy_df[symptom1_dummy_col]
symptom2_dummy_col = [col for col in get_dummy_df.columns if 'symptom2' in col]
symptom2_dummy = get_dummy_df[symptom2_dummy_col]
symptom3_dummy_col = [col for col in get_dummy_df.columns if 'symptom3' in col]
symptom3_dummy = get_dummy_df[symptom3_dummy_col]
symptom4_dummy_col = [col for col in get_dummy_df.columns if 'symptom4' in col]
symptom4_dummy = get_dummy_df[symptom4_dummy_col]
symptom5_dummy_col = [col for col in get_dummy_df.columns if 'symptom5' in col]
symptom5_dummy = get_dummy_df[symptom5_dummy_col]
symptom6_dummy_col = [col for col in get_dummy_df.columns if 'symptom6' in col]
symptom6_dummy = get_dummy_df[symptom6_dummy_col]

ls_dummy_vars = [
    location_dummy,
    country_dummy,
    gender_dummy,
    vis_wuhan_dummy,
    from_wuhan_dummy,
    symptom1_dummy,
    symptom2_dummy,
    symptom3_dummy,
    symptom4_dummy,
    symptom5_dummy,
    symptom6_dummy
]

ls_dummy_cols = [
    location_dummy_col,
    country_dummy_col,
    gender_dummy_col,
    vis_wuhan_dummy_col,
    from_wuhan_dummy_col,
    symptom1_dummy_col,
    symptom2_dummy_col,
    symptom3_dummy_col,
    symptom4_dummy_col,
    symptom5_dummy_col,
    symptom6_dummy_col
]

ls_dummy_vars_2 = list(map(lambda x: get_dummy_df[x], ls_dummy_cols))

#print(ls_dummy_vars_2)

#if functools.reduce(lambda x, y : x and y, map(lambda p, q: p == q,test_map,try_list), True): 
#    print ("The lists are the same") 
#else: 
#    print ("The lists are not the same")

#a = set(test_map)
#b = set(try_list)
 
#if a == b:
#    print("Lists l1 and l3 are equal")
#else:
#    print("Lists l1 and l3 are not equal")

check = map(lambda x, y: x.equals(y),ls_dummy_vars_2, ls_dummy_vars)

print(list(check))
