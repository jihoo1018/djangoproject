# Generated by Django 4.1.4 on 2022-12-28 08:42

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('categories', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Product',
            fields=[
                ('porduct_id', models.AutoField(primary_key=True, serialize=False)),
                ('name', models.TextField()),
                ('price', models.TextField()),
                ('image_url', models.TextField()),
                ('shop_category', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='categories.category')),
            ],
            options={
                'db_table': 'shop_product',
            },
        ),
    ]