from rest_framework import serializers
from blog.blog_users.models import BlogUser


class BlogUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = BlogUser
        fields = '__all__'

    def create(self,validated_data):
        return BlogUser.objects.create(**validated_data)

    def update(self,instace,valicated_data):
        BlogUser.objects.filter(pk=instace.id).update(**valicated_data)

    def delete(self,instance,valicated_data):
        pass
