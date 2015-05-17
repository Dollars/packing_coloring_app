import sys
import cProfile
from _lsprof import profiler_entry
# import pstats
import functools
import yappi
from inspect import signature
# import json
from time import process_time
# from packing_coloring.algorithms.solution import PackColSolution


class search_step_trace(object):

    __instances = {}

    def __init__(self, f):
        self.__f = f
        self.__name__ = f.__name__
        self.__numcalls = 0
        self.__elapsed_time = {'mean': 0.0, 'std': 0.0}
        self.__cumulative_time = 0.0
        search_step_trace.__instances[f] = self
        self.procceding = False

    def __call__(self, *args, **kwargs):
        yappi.stop()
        self.__numcalls += 1

        self.procceding = True
        self.start_time = process_time()

        result = self.__f(*args, **kwargs)

        elapsed = process_time() - self.start_time
        self.procceding = False

        self.__cumulative_time += elapsed
        old_mean = self.__elapsed_time['mean']
        old_std = self.__elapsed_time['std'] * (self.__numcalls - 1)

        delta = (elapsed - old_mean)
        new_mean = (old_mean + (delta/self.__numcalls))
        new_std = old_std + ((elapsed - old_mean) * (elapsed - new_mean))
        new_std = (new_std / self.__numcalls)
        self.__elapsed_time['mean'] = new_mean
        self.__elapsed_time['std'] = new_std

        yappi.start()

        return result

    def count(self):
        if self.procceding:
            return "error"
        return self.__numcalls

    def elapsed_time(self):
        if self.procceding:
            return "error"
        return self.__elapsed_time

    def cumulative_time(self):
        if self.procceding:
            return "error"
        return self.__cumulative_time

    def dump_vars(self):
        if self.procceding:
            return "error"
        trace = (self.count(),
                 self.elapsed_time(),
                 self.cumulative_time())
        return trace

    def clear_vars(self):
        self.__numcalls = 0
        self.__elapsed_time = {'mean': 0.0, 'std': 0.0}
        self.__cumulative_time = 0.0

    @staticmethod
    def set_trace(data):
        yappi.stop()
        for func in search_step_trace.__instances.values():
            name = func.__name__
            trace = data.get(name)
            if trace is not None:
                func.__numcalls = trace[0]
                func.__elapsed_time['mean'] = trace[1]['mean']
                func.__elapsed_time['std'] = trace[1]['std']
                func.__cumulative_time = trace[2]
        yappi.start()

    @staticmethod
    def dump_all():
        """Return a dict of {function: # of calls}
           for all registered functions."""
        yappi.stop()
        dump = {}
        for func, trace in search_step_trace.__instances.items():
            if trace.__numcalls > 0:
                stat = trace.dump_vars()
                dump[func.__name__] = stat
        yappi.start()
        return dump

    @staticmethod
    def clear_all():
        yappi.stop()
        for func, trace in search_step_trace.__instances.items():
            trace.clear_vars()
        yappi.start()

    @staticmethod
    def print_format():
        return '{0:<30} {1[0]:<10} {1[1][mean]:<10f} {1[1][std]:<10f} {1[2]:<10f}'

    @staticmethod
    def csv_format():
        return '{0}, {1[0]:d}, {1[1][mean]:f}, {1[1][std]:f}, {1[2]:f}'


def set_env(func):
    @functools.wraps(func)
    def echo_func(*args, **kwargs):
        sig = signature(func)
        par = sig.bind(*args, **kwargs)
        for kw, arg in par.kwargs.items():
            if kw == 'sol' and arg.record is not None:
                search_step_trace.set_trace(arg.record)

        return func(*args, **kwargs)
    return echo_func


class YProfiler(object):
    def __init__(self, *functions):
        self.enable_count = 0
        self.results = {}
        self.monitored = []

        for func in functions:
            self.add_function(func)

    def add_function(self, func):
        try:
            funcname = func.__name__
        except AttributeError:
            import warnings
            warnings.warn("Could not extract the name for the object %r" %
                          (func,))
            return

        if funcname not in self.monitored:
            self.monitored.append(funcname)

    def enable_by_count(self):
        if self.enable_count == 0:
            self.enable()
        self.enable_count += 1

    def disable_by_count(self):
        if self.enable_count > 0:
            self.enable_count -= 1
            if self.enable_count == 0:
                self.disable()

    def enable(self):
        yappi.start()

    def disable(self):
        yappi.stop()

    def wrap_function(self, func):
        try:
            funcname = func.__name__
        except AttributeError:
            import warnings
            warnings.warn("Could not extract the name for the object %r" %
                          (func,))
            return
        if funcname not in self.results:
            self.results[funcname] = {}

        @functools.wraps(func)
        def wrapper(*args, **kwds):
            self.enable_by_count()
            try:
                result = func(*args, **kwds)
            finally:
                self.disable_by_count()
                stats = yappi.get_func_stats()
                stats.strip_dirs()

                final_stats = []
                for entry in stats:
                    for name in self.monitored:
                        if name in entry.full_name:
                            entry.full_name = entry.full_name.split()[-1]
                            final_stats.append(entry)

                funcname = func.__name__
                probname = args[0].name
                if probname not in self.results[funcname]:
                    self.results[funcname][probname] = []

                self.results[funcname][probname].append((result.get_max_col(),
                                                        final_stats))
                yappi.clear_stats()
            return result
        return wrapper


class CProfiler():
    def __init__(self, *functions):
        self.enable_count = 0
        self.results = {}
        self.monitored = {}

        for func in functions:
            self.add_function(func)

        self.profiler = cProfile.Profile()

    def add_function(self, func):
        try:
            code = func.__code__
            funcname = func.__name__
        except AttributeError:
            import warnings
            warnings.warn("Could not extract the name for the object %r" %
                          (func,))
            return

        if funcname not in self.results:
            self.results[funcname] = {}

        if code not in self.monitored:
            self.monitored[code] = {}

    def enable_by_count(self):
        if self.enable_count == 0:
            self.enable()
        self.enable_count += 1

    def disable_by_count(self):
        if self.enable_count > 0:
            self.enable_count -= 1
            if self.enable_count == 0:
                self.disable()

    def enable(self):
        self.profiler.enable()

    def disable(self):
        self.profiler.disable()

    def wrap_function(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            self.enable_by_count()
            try:
                result = self.profiler.runcall(func, *args, **kwds)
            finally:
                self.disable_by_count()
                self.profiler.create_stats()
                stats = self.profiler.getstats()

                final_stats = []
                for entry in stats:
                    if entry[0] in self.monitored:
                        final_stats.append(entry)

                funcname = func.__name__
                probname = args[0].name
                if probname not in self.results[funcname]:
                    self.results[funcname][probname] = []

                self.results[funcname][probname].append((result.get_max_col(),
                                                        final_stats))
                self.profiler.clear()
            return result
        return wrapper


class Stats(object):
    def __init__(self, data):
        self.data = data

    def sort(self, crit="inlinetime"):
        if crit not in profiler_entry.__dict__:
            raise(ValueError, "Can't sort by %s" % crit)
        self.data.sort(lambda b, a: (getattr(a, crit) <
                                     getattr(b, crit)))
        for e in self.data:
            if e.calls:
                e.calls.sort(lambda b, a: (getattr(a, crit) <
                                           getattr(b, crit)))

    def pprint(self, top=None, file=None):
        """XXX docstring"""
        if file is None:
            file = sys.stdout
        d = self.data
        if top is not None:
            d = d[:top]
        cols = "% 12s %12s %11.4f %11.4f   %s\n"
        hcols = "% 12s %12s %12s %12s %s\n"
        # cols2 = "+%12s %12s %11.4f %11.4f +  %s\n"
        file.write(hcols % ("CallCount", "Recursive", "Total(ms)",
                            "Inline(ms)", "module:lineno(function)"))
        for e in d:
            file.write(cols % (e.callcount, e.reccallcount, e.totaltime,
                               e.inlinetime, label(e.code)))
            if e.calls:
                for se in e.calls:
                    file.write(cols % ("+%s" % se.callcount, se.reccallcount,
                                       se.totaltime, se.inlinetime,
                                       "+%s" % label(se.code)))

    def freeze(self):
        """Replace all references to code objects with string
        descriptions; this makes it possible to pickle the instance."""

        # this code is probably rather ickier than it needs to be!
        for i in range(len(self.data)):
            e = self.data[i]
            if not isinstance(e.code, str):
                self.data[i] = type(e)((label(e.code),) + e[1:])
            if e.calls:
                for j in range(len(e.calls)):
                    se = e.calls[j]
                    if not isinstance(se.code, str):
                        e.calls[j] = type(se)((label(se.code),) + se[1:])

_fn2mod = {}


def label(code):
    if isinstance(code, str):
        return code
    try:
        mname = _fn2mod[code.co_filename]
    except KeyError:
        for k, v in sys.modules.items():
            if v is None:
                continue
            if not hasattr(v, '__file__'):
                continue
            if not isinstance(v.__file__, str):
                continue
            if v.__file__.startswith(code.co_filename):
                mname = _fn2mod[code.co_filename] = k
                break
        else:
            mname = _fn2mod[code.co_filename] = '<%s>' % code.co_filename

    return '%s:%d(%s)' % (mname, code.co_firstlineno, code.co_name)
