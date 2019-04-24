from dask.distributed import Client, LocalCluster

class DaskClient():

    def __init__(self, ip_adress=None):
        if ip_adress is not None:
            #super(DaskClient, self).__init__(adress=ip_adress)
            self._client = Client(address=ip_adress) 
        else:
            self._client = Client()
            #super(DaskClient, self).__init__()
    def get_client(self):
        return self._client

    def set_func(self, func, kwargs={}):
        self.func = func
        self.kwargs = kwargs

    def wrapper(self, chunks):
        return [self.func(x, self.kwargs) for x in chunks]
    
    def get_futures(self, chunks):

        def wrapper(chunks):
            return [self.func(x, self.kwargs) for x in chunks]

        return self._client.map(wrapper, chunks)