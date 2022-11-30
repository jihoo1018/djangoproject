from django.db import models

from shop.categories.models import Category


class Product(models.Model):
    use_in_migration = True
    porduct_id = models.AutoField(primary_key=True)
    name = models.TextField()
    price = models.TextField()
    image_url = models.TextField()
    shop_category = models.ForeignKey(Category, on_delete=models.CASCADE)
    class Meta:
        db_table = "shop_product"
    def __str__(self):
        return f'{self.pk} {self.name} {self.price} {self.image_url}'
