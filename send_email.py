import smtplib
from email.mime.text import MIMEText

def email_sender(mail_host,mail_user,mail_pass,sender,receivers,context,content):
    #邮件内容设置
    message = MIMEText(content,'html','utf-8')
    #邮件主题       
    message['Subject'] = context
    #发送方信息
    message['From'] = sender 
    #接受方信息     
    message['To'] = receivers[0]  
    
    #登录并发送邮件
    try:
        smtpObj = smtplib.SMTP_SSL(mail_host, 465)
        #连接到服务器
        #smtpObj.ehlo(mail_host)
        #登录到服务器
        smtpObj.login(mail_user,mail_pass) 
        #发送
        smtpObj.sendmail(
            sender,receivers,message.as_string()) 
        #退出
        smtpObj.quit() 
        print('success')
    except smtplib.SMTPException as e:
        print('error',e) #打印错误