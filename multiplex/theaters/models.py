from django.db import models

from multiplex.cinemas.models import Cinema


class Theater(models.Model):
    use_in_migration = True
    theater_id = models.AutoField(primary_key=True)
    title = models.TextField()
    seat = models.TextField()
    multi_cinema = models.ForeignKey(Cinema, on_delete=models.CASCADE)
    class Meta:
        db_table = "multi_theater"
    def __str__(self):
        return f'{self.pk} {self.title} {self.seat}'
