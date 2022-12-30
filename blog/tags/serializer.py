from rest_framework import serializers
from blog.tags.models import Tag


class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = '__all__'

    def create(self, validated_data):
        return Tag.objects.create(**validated_data)

    def update(self, instace, valicated_data):
        Tag.objects.filter(pk=instace.id).update(**valicated_data)

    def delete(self, instance, valicated_data):
        pass

