from rest_framework import serializers
from shop.categories.models import Category


class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = '__all__'

    def create(self, validated_data):
        return Category.objects.create(**validated_data)

    def update(self, instace, valicated_data):
        Category.objects.filter(pk=instace.id).update(**valicated_data)

    def delete(self, instance, valicated_data):
        pass

