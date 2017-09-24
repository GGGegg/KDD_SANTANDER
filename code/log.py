import time
class Log(object):

	def __init__(self,outfile):
		super(Log,self).__init__()
		self.outfile = outfile
		self.outfile = open(self.outfile,'aw')

	def now(self):
		return time.striftime("%D  %H:%M:%S ")

	def log(self,message,time=True):
		print self.now() + message
		self.outfile.write(self.now() + message+"\n")

	def close(self):
		self.outfile.close()