from django.shortcuts import render, redirect, reverse, get_object_or_404

from django.http import HttpResponse, Http404
from mimetypes import guess_type

from IntelliCat.face_detection import *
from IntelliCat.predict_emotion import *
from IntelliCat.predict_race import *
from IntelliCat.models import *
from IntelliCat.cnnmodel import *
from IntelliCat.tests import *


# Create your views here.
def index(request):
    return render(request, 'index.html',{})

def gender(request):
    return render(request, 'gender.html',{})

def age(request):
    return render(request, 'age.html',{})

def emotion(request):
    return render(request, 'emotion.html',{})

def gender_guess(request):
    if request.method == 'GET':
        return render(request, 'gender.html',{})
    if request.FILES.get('photo'):
        new_pic = pic.objects.create(picture=request.FILES.get('photo'))
        results = analyze_picture(str(new_pic.picture.path))
        genders = []
        for result in results:
            if result == 0:
                genders.append("Male")
            else:
                genders.append("Female")
        num = len(genders)
        if(num == 1):
            genders = genders[0]
    return render(request, 'gender_result.html',{'genders':genders,
                                                 'currPic':new_pic,
                                                 'num': num})

def emotion_guess(request):
    if request.method == 'GET':
        return render(request, 'emotion.html',{})
    if request.FILES.get('photo'):
        new_pic = pic.objects.create(picture=request.FILES.get('photo'))
        results, logit = analyze_emotion(str(new_pic.picture.path))
        print(results)
        emotions = []
        for result in results:
            if result == 0:
                emotions.append("Angry")
            elif result == 1 or result == 2:
                emotions.append("Fear")
            elif result == 3:
                emotions.append("Sad")
            elif result == 4:
                emotions.append("Happy")
            elif result == 5:
                emotions.append("Surprise")
            else:
                emotions.append("Neutral")
        num = len(emotions)
        print(num)
    print(emotions)
    try:
        angry = logit[0][0] * 100
        print(angry)
        fear = logit[0][1] * 100 + logit[0][2] * 100
        print(fear)
        happy = logit[0][3] * 100
        print(happy)
        sad = logit[0][4] * 100
        print(sad)
        surprise = logit[0][5] * 100
        print(surprise)
        neutral = logit[0][6] * 100
        print(neutral)
        return render(request, 'emotion_result.html', {'emotions': emotions,
                                                       'currPic': new_pic,
                                                       'num': num,
                                                       'angry': angry,
                                                       'fear': fear,
                                                       'happy': happy,
                                                       'sad': sad,
                                                       'surprise': surprise,
                                                       'neutral': neutral})
    except:
        return render(request, 'emotion_result.html', {'emotions': emotions,
                                                       'currPic': new_pic,
                                                       'num': num,})

def race_guess(request):
    if request.method == 'GET':
        return render(request, 'age.html',{})
    if request.FILES.get('photo'):
        new_pic = pic.objects.create(picture=request.FILES.get('photo'))
        results, logit = analyze_race(str(new_pic.picture.path))
        print(results)
        race = []
        for result in results:
            # if result == 0:
            #     race.append("Hispanic")
            if result == 1:
                race.append("Caucasian")
            elif result == 2:
                race.append("Asian")
            else:
                race.append("African")
        num = len(race)
        print(num)
    print(race)
    try:
        caucasian = logit[0][0]
        print(caucasian)
        asian = logit[0][1]
        print(asian)
        african = logit[0][2]
        print(african)
        return render(request, 'race_result.html', {'race': race,
                                                    'currPic': new_pic,
                                                    'num': num,
                                                    # 'hispanic': hispanic,
                                                    'caucasian': caucasian,
                                                    'asian': asian,
                                                    'african': african,})
    except:
        return render(request, 'race_result.html', {'race': race,
                                                'currPic': new_pic,
                                                'num': num, })

def get_photo(request,id):
    currPic = get_object_or_404(pic,id=id)
    if not currPic.picture:
        return Http404
    content_type = guess_type(currPic.picture.name)
    return HttpResponse(currPic.picture,content_type=content_type)

def add_data(request,gender,flag,id):
    currPic = get_object_or_404(pic, id=id)
    if not currPic.picture:
        return Http404
    if gender == 'Male':
        if flag == 'T':
            put_data(str(currPic.picture.path),'male')
        else:
            put_data(str(currPic.picture.path), 'female')

    if gender == 'Female':
        if flag == 'F':
            put_data(str(currPic.picture.path),'male')
        else:
            put_data(str(currPic.picture.path), 'female')
    return render(request,'thanks.html', {'flag': flag})

