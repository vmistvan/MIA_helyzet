# helló, mi?
import torch
from torch import nn, optim
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

print("torch version:", torch.__version__)
print("pytorch ligthening version:", pl.__version__)

xor_input = [Variable(torch.Tensor([0, 0])),
             Variable(torch.Tensor([0, 1])),
             Variable(torch.Tensor([1, 0])),
             Variable(torch.Tensor([1, 1]))]

xor_target = [Variable(torch.Tensor([0])),
              Variable(torch.Tensor([1])),
              Variable(torch.Tensor([1])),
              Variable(torch.Tensor([0]))]

# A zip egy ravasz kis utasítás, összepattintja a bemeneteket, minden paraméter eleméből vesz egyet,
# és ezzel képez egy sort. Addig gyártja a sorokat, míg az egyik paraméter el nem fogy.
# itt egyszerre fogynak el, mivel 4 bemenetre 4 kimenet - ami itt az elvár target is - jut.
# végül lista típussá alakítja
xor_data = list(zip(xor_input, xor_target))
train_loader = DataLoader(xor_data, batch_size=1000)


# És íme a lehető legegyszerűbb modell:
# A kommenteket legalább egyszer szedd ki, hogy lásd az adatokat, tanulságos.
class XORModel(pl.LightningModule):
    def __init__(self):

        #  A Pytorch Lightning-ot örököltetjük, hogy hozzáférjünk az alap funkciókhoz is.
        super(XORModel,self).__init__()

        self.input_layer = nn.Linear(2, 4)
        self.output_layer = nn.Linear(4,1)
        self.sigmoid = nn.Sigmoid()
        # köszönjük MSE! de mindenképp nézd meg a többit is!
        self.loss = nn.MSELoss()

    # ezt még jobban át kell rágni, mindig elgondolkodtatott, hogy bár világos, hogy továbbítás,
    # de hogyan dönti el, hogy épp mely rétegek közt továbbít
    def forward(self, input):
        # print(\"INPUT:\", input.shape)
        x = self.input_layer(input)
        # print(\"FIRST:\", x.shape)
        x = self.sigmoid(x)
        # print(\"SECOND:\", x.shape)
        output = self.output_layer(x)
        # print(\"THIRD:\", output.shape)
        return output

    # Az optimalizáló függvény oldja meg, hogy a loss függvény szerint meghatározott hiba
    # a minimálisra csökkenjen a modell módosításával. érdemes rákeresni az adam-ra és megérteni, hogy mi
    # az lr azaz learning rate és mi az a momentum, ami átsegíti a helyi minimumok a modellt.
    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr=0.01)
        return optimizer

    # és itt az öszecsomagolt batch-ot szétkapja, az inputra ráküldi self-el magát a modellt
    # amit pedig eredményül kap, összeveti az xor_target - elvárt eredménnyel
    # így visstzadja a loss-t, ami meg a fenti optimalizáló próbál a neuronok állításával csökkenteni. repeat!
    def training_step(self, batch, batch_idx):
        xor_input, xor_target = batch
        # print(\"XOR INPUT:\", xor_input.shape)
        # print(\"XOR TARGET:\", xor_target.shape)
        outputs = self(xor_input)
        # print(\"XOR OUTPUT:\", outputs.shape)
        loss = self.loss(outputs, xor_target)
        return loss


# az alábbiakat nem teljesen így csinálom, meg kell érteni mit akar a callback-kal,
# és én megadtam egy direkt nevet is neki, de eredetileg a kikommentezett checkpoint kezelés volt benne.
# a test változót az alábbi példában nem használja látszólag semmire.
checkpoint_callback = ModelCheckpoint()
model = XORModel()

trainer = pl.Trainer(max_epochs=600, callbacks=[checkpoint_callback])

trainer.fit(model, train_dataloaders=train_loader)

trainer.save_checkpoint("XOR_model.pt")

print(checkpoint_callback.best_model_path)

# első az eredeti, de controllfreakbe mentem át, és megadtam én.
# train_model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
train_model = XORModel.load_from_checkpoint("XOR_model.pt")
test = torch.utils.data.DataLoader(xor_input, batch_size=1)
for val in xor_input:
    _ = train_model(val)
    print([int(val[0]),int(val[1])], int(_.round()))

