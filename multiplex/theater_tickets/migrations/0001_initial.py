# Generated by Django 4.1.4 on 2022-12-28 08:42

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('theaters', '0001_initial'),
        ('movie_users', '0001_initial'),
        ('showtimes', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='TheaterTicket',
            fields=[
                ('theater_ticket_id', models.AutoField(primary_key=True, serialize=False)),
                ('x', models.IntegerField()),
                ('y', models.IntegerField()),
                ('multi_showtime', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='showtimes.showtime')),
                ('multi_theater', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='theaters.theater')),
                ('multi_user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='movie_users.movieuser')),
            ],
            options={
                'db_table': 'multi_theater_tickets',
            },
        ),
    ]
