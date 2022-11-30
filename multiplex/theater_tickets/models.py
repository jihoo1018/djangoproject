from django.db import models

from multiplex.movie_users.models import MovieUser
from multiplex.showtimes.models import Showtime
from multiplex.theaters.models import Theater


class TheaterTicket(models.Model):
    use_in_migration = True
    theater_ticket_id = models.AutoField(primary_key=True)
    x = models.IntegerField()
    y = models.IntegerField()
    multi_showtime = models.ForeignKey(Showtime, on_delete=models.CASCADE)
    multi_theater = models.ForeignKey(Theater, on_delete=models.CASCADE)
    multi_user = models.ForeignKey(MovieUser,on_delete=models.CASCADE)
    class Meta:
        db_table = "multi_theater_tickets"
    def __str__(self):
        return f'{self.pk} {self.x} {self.y}'
