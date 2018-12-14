from django.db import models

# Create your models here.
class pic(models.Model):
    picture = models.ImageField(upload_to="data", default='data/default.png',blank=True)
    gender = models.CharField(max_length=10,)
    age = models.IntegerField(verbose_name='age', default=0)
    emotion = models.CharField(max_length=10,)

    def __str__(self):
        return self.picture