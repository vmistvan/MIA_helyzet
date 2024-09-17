##### 181. oldaltól - eddig jutottam az értelmezésben
import accelerate.accelerator
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
from torch.autograd import Variable

import pytorch_lightning as pl
from pytorch_lightning import seed_everything, LightningModule, Trainer
import pandas as pd
pd.options.mode.chained_assignment = None # default = warn - a példa másmilyen
import matplotlib.pyplot as plt
import datetime

torch.set_float32_matmul_precision('medium')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Set device to "cuda" if it's available otherwise default to "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

# sns.set(rc={'figure.figure':(25,8)})
# töltsük be: Metro_Interstate_Traffic_Volume.csv
df_traffic = pd.read_csv('./Metro_Interstate_Traffic_Volume.csv', parse_dates=['date_time'], index_col="date_time")

print ("Oszlopok", df_traffic.dtypes)


# feltáró adatelemzés, ha megvan, az ismételt futtatáskhoz kikommentezni
# print(df_traffic.head())

# print("sorok száma: ", df_traffic.shape[0])
# print("oszlopok száma: ", df_traffic.shape[1])

# print (df_traffic.weather_main.value_counts())
# print (df_traffic.holiday.value_counts())
# print (df_traffic[df_traffic.index.duplicated()])
# feltáró adatelemzés vége

df_traffic = df_traffic[~df_traffic.index.duplicated(keep='last')]
print("sorok száma (duplikáltak eldobva): ", df_traffic.shape[0])

# grafikon megjelenítése. ismételt futtatáskor ki lehet szedni, csak debughoz
# de egyszer szedd ki a kommentből, mert annyira szép!
# plt.xticks(
#  rotation=90,
#  horizontalalignment='right',
#  fontweight='light',
#  fontsize='x-small'
# )
#
# plt.title("Time series: traffic volume")
# sns.lineplot(x=df_traffic.index, y=df_traffic["traffic_volume"])
#
# plt.show()

date_range = pd.date_range('2012-10-02 09:00:00', '2018-09-30 23:00:00', freq='1h')
df_dummy = pd.DataFrame(np.random.randint(1, 20, (date_range.shape[0], 1)))
df_dummy.index = date_range # set index
df_missing = df_traffic

#check for missing datetimeindex values based on reference index (with all values)
missing_hours = df_dummy.index[~df_dummy.index.isin(df_missing.index)]

print("Hiányzó órák:")
print(missing_hours)

print(df_traffic['temp'].describe())

df_traffic['temp']=df_traffic['temp'].replace(0,df_traffic['temp'].median())

print("Javítva a 0 kelvinnel:")
print(df_traffic['temp'].describe())

# miután az indexet meghatároztuk, és dátum szerint értelmeztük, elég
# könnyű kizárólag egyes éveket
# a megújított táblába másolni
df_traffic = df_traffic[df_traffic.index.year.isin([2016,2017,2018])].copy()



# Filling missing values using backfill and interpolation methods

# az alábbi sok komment jelzi, hogy mekkora epic csatát vívtam a teljesen rossz eredeti példával.

# deprecated, alább korszerűbb: df_traffic = pd.concat([df_traffic.select_dtypes(include=['object']).fillna(method='backfill'), df_traffic.select_dtypes(include=['float']).interpolate()], axis=1)
# a fenti sor tehát outdated, csak Uncle Bob kedvéért hagytam benne, hogy lássa, nem vagyok professional.
# a sor átírása kevés bfill-re, annak paramétere axis=1 - amit a példa nem tartalmaz így, de máshogy nem nem tűnt el minden hiányzó érték
# innen hiányzott egy in érték, amit dettó bfillel oldottam meg a csoport nagyobb dicsőségére
# df_traffic = pd.concat([df_traffic.select_dtypes(include=['object']).bfill(axis="rows"), df_traffic.select_dtypes(include=['int64']).bfill(axis="rows"), df_traffic.select_dtypes(include=['float']).interpolate()], axis=0)
# df_traffic = pd.concat([df_traffic.select_dtypes(include=['object']).ffill(axis="rows", limit=23), df_traffic.select_dtypes(include=['int64']).bfill(axis="columns"), df_traffic.select_dtypes(include=['float']).interpolate()], axis="columns")
# print(df_traffic.shape)
# df_traffic = pd.concat([df_traffic.select_dtypes(include=['object']).fillna(method='ffill', limit=1), df_traffic.select_dtypes(include=['int64']).fillna(method='ffill', limit=1), df_traffic.select_dtypes(include=['float']).interpolate()], axis=1)
df_traffic = pd.concat([df_traffic['holiday'].fillna(value="None"), df_traffic['weather_main'].fillna(method='bfill'), df_traffic.select_dtypes(include=['int64']).fillna(method='ffill', limit=1), df_traffic.select_dtypes(include=['float']).interpolate()], axis=1)

print ("Oszlopok", df_traffic.dtypes)

print(df_traffic.isna().sum())
# print("hol vannak a sorok? ", df_traffic.columns)
# print ("hogy kerül a köd az ünnepekbe??", df_traffic.holiday.value_counts())
# ez egy ravasz funkció. ahhelyett, hogy lenne 1 oszlop sok értékkel, lesz kismillió oszlop a megadott oszlopból, viszont érték csak igaz/hamis lehet.
# mint az egyszeri választó, aki nem írhatja be, hogy mire szavaz, hanem bejelöli a választottját. Tehát egy igaz, többi hamis.
# ezt érdemes a leírásból értelmezni, mert eléggé érhetetlen máshogy, amit akár a head is ad.
# észre kell venni, hogy itt két kategóriában csinál ilyen oszlopválasztós dummies parádét: holiday és weather_main lesz szétcsapva.

df_traffic = pd.get_dummies(df_traffic, columns = ['holiday', 'weather_main'], drop_first=True)

# aki okos, az nem a következő sorban droppol, hanem eleve át se veszi az új táblába, így már megspórol sok küzdelmet
# df_traffic.drop('weather_description', axis=1, inplace=True)
df_traffic.to_csv('out.csv', index=True)


print(df_traffic.shape)
print(df_traffic.head())
print(df_traffic.columns)

print("Minden ekkor kezdődött:", df_traffic.index.min())
print("Egyszer minden végetér:", df_traffic.index.max())

# Tréning:
df_traffic_train = df_traffic.loc[:datetime.datetime(year=2017,month=12,day=31,hour=23)]

print("Total number of row in train dataset:", df_traffic_train.shape[0])

print("Train dataset start date :",df_traffic_train.index.min())
print("Train dataset end date:",df_traffic_train.index.max())

# Validálás:
df_traffic_val =df_traffic.loc[datetime.datetime(year=2018,month=1,day=1, hour=0):datetime.datetime(year=2018,month=6,day=30,hour=23)]

print("Total number of row in validate dataset:", df_traffic_val.shape[0])

print("Validate dataset start date :",df_traffic_val.index.min())
print("Validate dataset end date:",df_traffic_val.index.max())

# Tesztelés:
df_traffic_test = df_traffic.loc[datetime.datetime(year=2018,month=7,day=1,hour=0):]

print("Total number of row in test dataset:", df_traffic_test.shape[0])

print("Validate dataset start date :", df_traffic_test.index.min())
print("Validate dataset end date:",df_traffic_test.index.max())

#create scalers
temp_scaler = MinMaxScaler()
rain_scaler = MinMaxScaler()
snow_scaler = MinMaxScaler()
cloud_scaler = MinMaxScaler()
volume_scaler = MinMaxScaler()

#Create transformers
temp_scaler_transformer = temp_scaler.fit(df_traffic_train[['temp']])
rain_scaler_transformer = rain_scaler.fit(df_traffic_train[['rain_1h']])
snow_scaler_transformer = snow_scaler.fit(df_traffic_train[['snow_1h']])
cloud_scaler_transformer = cloud_scaler.fit(df_traffic_train[['clouds_all']])
volume_scaler_transformer = volume_scaler.fit(df_traffic_train[['traffic_volume']])

df_traffic_train["temp"]= temp_scaler_transformer.transform(df_traffic_train[['temp']])
df_traffic_train["rain_1h"]= rain_scaler_transformer.transform(df_traffic_train[['rain_1h']])
df_traffic_train["snow_1h"]= snow_scaler_transformer.transform(df_traffic_train[['snow_1h']])
df_traffic_train["clouds_all"]=cloud_scaler_transformer.transform(df_traffic_train[['clouds_all']])
df_traffic_train["traffic_volume"]=volume_scaler_transformer.transform(df_traffic_train[['traffic_volume']])

df_traffic_val["temp"] = temp_scaler_transformer.transform(df_traffic_val[['temp']])
df_traffic_val["rain_1h"] = rain_scaler_transformer.transform(df_traffic_val[['rain_1h']])
df_traffic_val["snow_1h"] = snow_scaler_transformer.transform(df_traffic_val[['snow_1h']])
df_traffic_val["clouds_all"] = cloud_scaler_transformer.transform(df_traffic_val[['clouds_all']])
df_traffic_val["traffic_volume"] = volume_scaler_transformer.transform(df_traffic_val[['traffic_volume']])

df_traffic_test["temp"] = temp_scaler_transformer.transform(df_traffic_test[['temp']])
df_traffic_test["rain_1h"] = rain_scaler_transformer.transform(df_traffic_test[['rain_1h']])
df_traffic_test["snow_1h"] = snow_scaler_transformer.transform(df_traffic_test[['snow_1h']])
df_traffic_test["clouds_all"] = cloud_scaler_transformer.transform(df_traffic_test[['clouds_all']])
df_traffic_test["traffic_volume"] = volume_scaler_transformer.transform(df_traffic_test[['traffic_volume']])


class TrafficVolumeDataset(torch.utils.data.Dataset):
    def __init__(self, train=False, validate=False, test=False, window_size=480):

        # STEP1: Load the data
        self.df_traffic_train = df_traffic_train
        self.df_traffic_val = df_traffic_val
        self.df_traffic_test = df_traffic_test

        # STEP2: Creating Features
        if train:  # process train dataset
            features = self.df_traffic_train
            target = self.df_traffic_train.traffic_volume
        elif validate:  # process validate dataset
            features = self.df_traffic_val
            target = self.df_traffic_val.traffic_volume
        else:  # process test dataset
            features = self.df_traffic_test
            target = self.df_traffic_test.traffic_volume

        # STEP3: Create windows/sequencing
        self.x, self.y = [], []
        for i in range(len(features) - window_size):
            v = features.iloc[i:(i + window_size)].values
            self.x.append(v)
            self.y.append(target.iloc[i + window_size])

        # STEP4: Calculate length of dataset
        self.num_sample = len(self.x)

    def __getitem__(self, index):
        x = self.x[index].astype(np.float32)
        y = self.y[index].astype(np.float32)
        return x, y

    def __len__(self):
        # returns the total number of records for data set
        return self.num_sample


traffic_volume =  TrafficVolumeDataset(test=True)


#let's loop it over single iteration and print the shape and also data
for i, (features,targets) in enumerate(traffic_volume):
  print("Size of the features",features.shape)
  print("Printing features:\n", features)
  print("Printing targets:\n", targets)
  break


class TrafficVolumePrediction(pl.LightningModule):
    def __init__(self, input_size=26, output_size=1, hidden_dim=10, n_layers=2, window_size=480):
        """
        input_size: Number of features in the input
        hidden_dim: number of hidden layers
        n_layers: number of RNN to stack over each other
        output_size: number of items to be outputted
        """
        super(TrafficVolumePrediction, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.TrafficVolumePrediction = nn.LSTM(input_size, hidden_dim, n_layers, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_dim * window_size, output_size)

        self.loss = nn.MSELoss()

        self.learning_rate = 0.001

        # self.relu = nn.ReLU()

    def forward(self, x):
        # x=x.to("cuda")

        batch_size = x.size(0)
        hidden = self.get_hidden(batch_size)

        # print("cuda", torch.cuda.memory_summary(0))
        out, hidden = self.TrafficVolumePrediction(x, hidden)
        # out = self.relu(out)
        out =out.reshape(out.shape[0], -1)
        out = self.fc(out)

        return out


    def get_hidden(self, batch_size):
        # hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        cell_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        hidden = (hidden_state, cell_state)
        return hidden

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr = self.learning_rate)
        # optimizer = optim.RMSprop(params=params, lr = 0.001)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        features, targets = train_batch
        output = self(features)
        output = output.view(-1)
        loss = self.loss(output, targets)
        self.log('train_loss', loss, prog_bar=True)
        return {"loss": loss}

    def train_dataloader(self):
        traffic_volume_train =  TrafficVolumeDataset(train=True)
        train_dataloader = torch.utils.data.DataLoader(traffic_volume_train, batch_size=50)
        return train_dataloader

    def validation_step(self, val_batch, batch_idx):
        features, targets = val_batch
        output = self(features)
        output = output.view(-1)
        loss = self.loss(output, targets)
        self.log('val_loss', loss, prog_bar=True)

    def val_dataloader(self):
        traffic_volume_val =  TrafficVolumeDataset(validate=True)
        val_dataloader = torch.utils.data.DataLoader(traffic_volume_val, batch_size=50)
        return val_dataloader




seed_everything(10)
model = TrafficVolumePrediction()
model.to(device)

trainer = pl.Trainer(max_epochs=140, accelerator='gpu', devices=-1)

# mivel az alábbi sorok elől sohase vettem ki a kommentet, olyan pillanatra tartogatom őket
# amikor már teljesen világos lesz a működése a modellnek, és valamiért kell a fájdalom.
# Run learning rate finder
#lr_finder = trainer.tuner.lr_find(model, min_lr=1e-04, max_lr=1, num_training=30)
# Pick point based on plot, or get suggestion
#new_lr = lr_finder.suggestion()
#print("Suggested Learning Rate is :", new_lr)
# update hparams of the model
#model.hparams.lr = new_lr

trainer.fit(model)

# és itt jönne a modell igazi használata, valami önálló prediction, de azt majd egy boldogabb világban
# nem volt a példa része, és ettől a modelltől az is szép, hogy lefut, és konvergál.
