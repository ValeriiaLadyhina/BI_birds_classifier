from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from io import BytesIO
import cv2
import numpy as np
import torch
from torchvision import models, datasets, transforms
import torch.nn as nn
import pandas as pd


def start(update, context):
    update.message.reply_text('Welcome to What Bird Do I See? bot.\n I classify bird images '
                              'using ResNet18 model. To start work, please send me a snap of a 🐦 you are interested in,'
                              ' otherwise, if  you need any help please type "/help"')


def description(update, context):
    update.message.reply_text('To start analysis of a species of your bird please sent to the chat your image.')


def help(update, context):
    update.message.reply_text('''
    /start - start conversation\n/help - help\n/description - explanation of what bot is expecting to get from 
    you to start analysis
    ''')


def handle_message(update, context):
    update.message.reply_text('I am sorry, I work only with images. Please send me a picture of a bird that '
                              'you are interested in.')



def handle_photo(update: Update, context: CallbackContext):
    update.message.reply_text('I will now try to identify what is the bird on your snap, please wait :)')
    birds = pd.read_csv('birds latin names.csv')
    chat_id = update.message.chat_id
    classes_names = birds['class']
    latin_names = birds['SCIENTIFIC NAME']
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f =  BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(size=(224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.4704, 0.4669, 0.3898], [0.2392, 0.2329, 0.2546])
                                    ])
    img = transform(img)
    image_to_be_analysed = img.unsqueeze(0)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 400)
    #model.load_state_dict(torch.load('ResNet18_fine_tune_best.pth'))
    model.load_state_dict(torch.load('ResNet18_fine_tune_final.pth'))

    with torch.no_grad():
        model.eval()
        probs = model(image_to_be_analysed)
    probs = nn.functional.softmax(probs.data, dim=1)
    top_probs = torch.topk(probs, 1)
    for pred_labels, values in zip(top_probs.indices, top_probs.values):
        for pred_label, pred_value in zip(pred_labels, values):
            class_name = f"{classes_names.iloc[pred_label.item()]}"
            latin_name = f"{latin_names.iloc[pred_label.item()]}"
            probability = float(f"{pred_value.item() * 100.0:.3f}")
    #update.message.reply_text('I think that the bird on the picture looks similar to the next one on the image:')
    #photo ='/Users/vana0005/Bird_Classifier/archive/train/'+class_name+'/001.jpg'
    #update.message.reply_photo(photo = photo)
    if probability >= 50:
        update.message.reply_text('I think that the bird on the picture is ' + class_name +
                                  ' with official latin name: ' +
                              latin_name + ', with probability ' + str(probability) + '%, but I can be mistaken 🤪 and I '
                                                                                 'hope you had a bit of fun.')
        update.message.reply_text('If you want to find a type of another beautiful birdy please send '
                                  'me a new image. :)')
    else:
        update.message.reply_text('I am sorry, I do not know what it is, can you please try another image? '
                                  'Or maybe it was not a bird?')




def main():

    TOKEN = '5377329699:AAFtiC7pV90TOJlBXAy18mywU2o1Zcgqh5s'
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('help', help))
    dp.add_handler(CommandHandler('description', description))
    dp.add_handler(MessageHandler(Filters.text, handle_message))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))


    updater.start_polling()
    updater.idle()



if __name__ == '__main__':
    main()
