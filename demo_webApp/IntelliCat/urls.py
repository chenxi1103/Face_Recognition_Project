#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Chenxi Li on 2018-12-05
from django.urls import path
from IntelliCat import views
from django.conf.urls import url
from demo import settings
from django.views.static import serve
from django.conf.urls.static import static
urlpatterns = [
    path('', views.index, name='index'),
    path('gender', views.gender, name='gender'),
    path('age', views.age, name='age'),
    path('emotions', views.emotion, name='emotions'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)