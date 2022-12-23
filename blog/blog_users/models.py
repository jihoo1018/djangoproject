from django.db import models

class BlogUser(models.Model):
    use_in_migration = True
    blog_userid = models.AutoField(primary_key=True)
    email = models.CharField(max_length=20)
    nickname = models.CharField(max_length=20)
    password = models.CharField(max_length=20)

    class Meta:
        db_table = "blog_users"
    def __str__(self):
        return f'{self.pk} {self.email} {self.nickname} {self.password}'
