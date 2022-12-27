# Generated by Django 4.1.4 on 2022-12-23 05:25

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('movies', '0001_initial'),
        ('cinemas', '0001_initial'),
        ('theaters', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Showtime',
            fields=[
                ('showtime_id', models.AutoField(primary_key=True, serialize=False)),
                ('start_time', models.DateTimeField()),
                ('end_time', models.DateTimeField()),
                ('multi_cinema', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='cinemas.cinema')),
                ('multi_movie', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='movies.movie')),
                ('multi_theater', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='theaters.theater')),
            ],
            options={
                'db_table': 'multi_showtime',
            },
        ),
    ]
