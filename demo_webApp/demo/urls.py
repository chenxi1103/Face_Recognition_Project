"""demo URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from IntelliCat.views import *
from django.conf.urls import url
from demo import settings
from django.views.static import serve
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
    path('gender', gender, name='gender'),
    path('gender_guess',gender_guess,name='gender_guess'),
    path('emotion_guess',emotion_guess,name='emotion_guess'),
    path('race_guess',race_guess,name='race_guess'),
    path('race', age, name='age'),
    path('emotions', emotion, name='emotions'),
    url(r'^photo/(?P<id>\d+)$', get_photo, name='photo'),
    url(r'^add_data/(?P<gender>.+)/(?P<flag>.+)/(?P<id>\d+)/$',add_data, name='add_data'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
