# Generated by Django 4.1.3 on 2022-11-30 03:56

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='MovieUser',
            fields=[
                ('movieuser_id', models.AutoField(primary_key=True, serialize=False)),
                ('email', models.TextField()),
                ('nickname', models.TextField()),
                ('password', models.TextField()),
                ('age', models.TextField()),
            ],
            options={
                'db_table': 'multi_movie_user',
            },
        ),
    ]
