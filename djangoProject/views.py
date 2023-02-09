from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse


def index(request):
    # latest_question_list = Question.objects.order_by('-pub_date')[:5]
    context = {}
    return render(request, 'index.html', context)