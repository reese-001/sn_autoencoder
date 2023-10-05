#%%
import pandas as pd
from servicenow_utils import get_incidents
import preprocessing
import autoencoder
import visualize

from sklearn.feature_extraction.text import TfidfVectorizer

password  = ''
user_name = ''
url       = 'https://.service-now.com'
endpoint  = '/api/bebup/incident/query/list'


# %%

## check to see if data/incidents.csv exists
## if it does, then read it in
## if it does not, then get the data from servicenow
try:
    df = pd.read_csv('data/incidents.csv')
except:
    df = get_incidents.get_incidents(password=password, 
                                    user_name=user_name, 
                                    url=url, 
                                    endpoint=endpoint,
                                    current_day='2023-10-01', 
                                    end_day='2023-10-04')

    df.to_csv('data/incidents.csv')
    
try:
    df['assignment_group'] = df['assignment_group'].apply(lambda x: x.split("'name': '")[-1].split("'}")[0])
except:
    pass

# %%
X_combined_train, X_combined_test, train_indices_combined, test_indices_combined = preprocessing.preprocess(df)
    
# %%
model = autoencoder.train(X_combined_train)
recon_error = autoencoder.reconstruction_error(model, X_combined_test)

# %%
df_anomalies = visualize.with_threshold(recon_error, 99.5, df, test_indices_combined)  # Using 95 as the percentile for threshold
print(df_anomalies)
# %%
df_anomalies.to_csv('data/anomomies.csv')
# %%



