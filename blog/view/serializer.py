from rest_framework import serializers
from blog.view.models import View


class BlogViewSerializer(serializers.ModelSerializer):
    class Meta:
        model = View
        fields = '__all__'

    def create(self, validated_data):
        return View.objects.create(**validated_data)

    def update(self, instace, valicated_data):
        View.objects.filter(pk=instace.id).update(**valicated_data)

    def delete(self, instance, valicated_data):
        pass

