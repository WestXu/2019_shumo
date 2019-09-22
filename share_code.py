import functools
import json
from pathlib import Path

import gurobipy
import methodtools
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lazy_object_proxy.utils import cached_property
from tqdm.auto import tqdm

pd.options.display.max_rows = 10


def football_score(x0, y0, z0, A, B):
    to_A_distance = np.sqrt(sum((np.array([x0, y0, z0]) - A) ** 2))
    to_B_distance = np.sqrt(sum((np.array([x0, y0, z0]) - B) ** 2))
    return to_A_distance + to_B_distance


def barrel_score(x0, y0, z0, A, B):
    M = np.array([x0, y0, z0])
    S = A - B
    D = np.linalg.norm(np.cross((M - A), S)) / np.linalg.norm(S)
    return D


def dumbbell_score(x0, y0, z0, A, B):
    to_A_distance = np.sqrt(sum((np.array([x0, y0, z0]) - A) ** 2))
    to_B_distance = np.sqrt(sum((np.array([x0, y0, z0]) - B) ** 2))
    return min(to_A_distance, to_B_distance) / 10 + barrel_score(x0, y0, z0, A, B)


class Solver:
    def __init__(
        self, which_dataset=1, subsample_how=None, subsample_rows=None, max_step=None
    ):
        assert which_dataset in (1, 2)
        self.which_dataset = which_dataset
        if which_dataset == 1:
            self.alpha1 = 25
            self.alpha2 = 15
            self.beta1 = 20
            self.beta2 = 25
            self.theta = 30
            self.delta = 0.001
            self.data_filename = "附件1：数据集1-终稿.xlsx"
        else:
            self.alpha1 = 20
            self.alpha2 = 10
            self.beta1 = 15
            self.beta2 = 20
            self.theta = 20
            self.delta = 0.001
            self.data_filename = "附件2：数据集2-终稿.xlsx"

        self.subsample_how = subsample_how
        self.subsample_rows = subsample_rows
        self.max_step = max_step

        self.basic_model_inited = False

    @cached_property
    def raw_df(self):
        '''未经任何筛选的df'''
        df = (
            pd.read_excel(
                Path('questions') / '2019年中国研究生数学建模竞赛F题' / self.data_filename,
                skiprows=1,
                index_col=0,
            )
            .replace('A 点', 'A点')
            .rename(columns=lambda _: _.replace("（单位: m）", "").replace("（单位:m）", ""))
            .rename(columns={'校正点标记': '校正点类型'})
            .rename_axis('编号')
        )

        if self.subsample_how is None:
            return df

        A_coordinates = df.iloc[0][['X坐标', 'Y坐标', 'Z坐标']].astype(float).values
        B_coordinates = df.iloc[-1][['X坐标', 'Y坐标', 'Z坐标']].astype(float).values

        if self.subsample_how == 'football':
            '''筛选一个以AB为焦点的橄榄球内的点'''
            score_fucntion = football_score
        if self.subsample_how == 'barrel':
            '''筛选一个以AB所连直线为中心轴的圆柱体内的点'''
            score_fucntion = barrel_score
        if self.subsample_how == 'dumbbell':
            '''筛选一个以AB所连直线为中心轴的哑铃型内的点'''
            score_fucntion = dumbbell_score

        df['subsample_score'] = df.apply(
            lambda _: score_fucntion(
                _['X坐标'], _['Y坐标'], _['Z坐标'], A_coordinates, B_coordinates
            ),
            axis=1,
        )
        df['in_subsample'] = (
            df.subsample_score
            < df.subsample_score.sort_values().iloc[self.subsample_rows]
        )

        return df

    @cached_property
    def df(self):
        '''筛选后的df'''
        df = self.raw_df.copy()

        if self.subsample_rows is not None:
            df = df.loc[lambda _: _.in_subsample].copy()

        if self.subsample_rows:
            df = pd.concat(
                [
                    df.iloc[[0]],
                    df.loc[
                        lambda _: (_.in_subsample) & (~(_['校正点类型'].isin(['A点', 'B点'])))
                    ],
                    df.iloc[[-1]],
                ]
            )
        # df = df.reset_index(drop=True)
        return df

    @cached_property
    def N(self):
        '''总步数+1,包括后面停止的步数'''
        if self.max_step is None:
            return len(self.df)
        else:
            return self.max_step + 1

    @cached_property
    def I(self):
        '''点数'''
        return len(self.df)

    def get_init_fig(self, only_sample=True, color='校正点类型'):

        fig = px.scatter_3d(
            (self.df if only_sample else self.raw_df)
            .reset_index()
            .assign(in_subsample=lambda _: _.in_subsample.astype(str)),
            x="X坐标",
            y="Y坐标",
            z="Z坐标",
            color=color,
            hover_data=['编号'],
        )

        fig.update_traces(marker_size=3)
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[0 * 1e3, 100 * 1e3]),
                yaxis=dict(range=[0 * 1e3, 100 * 1e3]),
                zaxis=dict(range=[-50 * 1e3, 50 * 1e3]),
            )
        )
        fig.update_layout(template='plotly_white')
        return fig

    @methodtools.lru_cache(maxsize=None)
    def is_horizontal(self, j):
        return self.df.iloc[j, '校正点类型']

    @methodtools.lru_cache(maxsize=None)
    def is_vertical(self, j):
        return bool(not self.df.iloc[j, '校正点类型'])

    @methodtools.lru_cache(maxsize=None)
    def get_adjust_type(self, j):
        return {1: 'veltical', 0: 'horizontal'}[self.df.iloc[j, '校正点类型']]

    @methodtools.lru_cache(maxsize=None)
    def get_distance(self, i, j):
        assert j != 0
        if i == j:
            return 0
        vs = self.df.values[[i, j], :3]
        return np.sqrt(sum((vs[1] - vs[0]) ** 2))

    def get_var_res(self, var_name='F', only_val=None, not_val=None):
        '''获取计算结果中变量的值'''
        var_res = [
            (v.Varname, v.x)
            for v in self.model.getVars()
            if v.varName.split('[')[0] == var_name
        ]
        if only_val is not None:
            var_res = [_ for _ in var_res if round(_[1]) == only_val]
        if not_val is not None:
            var_res = [_ for _ in var_res if round(_[1]) != not_val]
        return var_res

    def init_basic_model(self):
        N = self.N
        I = self.I
        delta = self.delta

        print('Initialize model')
        model = gurobipy.Model()

        print('Fni表示第n步是否走到编号为i的点,n从1开始')
        F = model.addVars(range(1, N), range(1, I), vtype=gurobipy.GRB.BINARY, name='F')
        self.F = F

        C = model.addVars(range(1, N), vtype=gurobipy.GRB.INTEGER, name='C')
        self.C = C
        for n in tqdm(range(1, N), desc='C[n]表示第n步走到哪个点'):
            for i in range(1, I):
                model.addConstr((F[n, i] == 1) >> (C[n] == i))
                # NOTE:这里还缺一个反向的约束 就是F ==0的时候,Cn不为i,但是gorubi不能写不等于,这是problem1

        for n in tqdm(range(1, N), desc='每一步最多走一个点'):
            model.addConstr(F.sum(n, '*') <= 1)

        for i in tqdm(range(1, I), desc='每个点最多被走一次'):
            model.addConstr(F.sum('*', i) <= 1)

        print('一定会走到B点')
        model.addConstr(F.sum('*', I - 1) == 1)

        Q = model.addVars(range(1, N), vtype=gurobipy.GRB.BINARY, name='Q')
        self.Q = Q
        for n in tqdm(range(1, N), desc='Qn表示第n步是否还在走'):
            model.addConstr(Q[n] == F.sum(n, '*'))

            # NOTE:这里解决上面提到的problem1
            model.addConstr((Q[n] == 0) >> (C[n] == 0))

        for n in tqdm(range(1, N), desc='n步如果不在走了,那之后也不在走了'):
            model.addConstrs((Q[n] == 0) >> (Q[m] == 0) for m in range(n + 1, N))

        for n in tqdm(range(1, N), desc='n步如果走到了B点,之后就不在走了'):
            model.addConstrs((F[n, I - 1] == 1) >> (Q[m] == 0) for m in range(n + 1, N))

        print('Bn表示第n步走的距离')
        B = model.addVars(range(1, N), vtype=gurobipy.GRB.INTEGER, name="B")
        self.B = B
        # 其中第一步的距离需要单独计算
        for j in tqdm(range(1, I), desc='B1需要单独计算'):
            model.addGenConstrIndicator(
                F[1, j],
                True,
                B[1],
                gurobipy.GRB.EQUAL,
                round(self.get_distance(0, j) * delta),
            )

        for n in tqdm(range(2, N), desc='添加Bn约束'):  # 不需要包括第一步,第二步之后就包括了所有信息
            model.addConstr(B[n] >= 0)
            model.addConstr(
                B[n]
                == gurobipy.quicksum(
                    [
                        F[n - 1, i] * F[n, j] * round(self.get_distance(i, j) * delta)
                        for i in range(1, I)
                        for j in range(1, I)
                    ]
                )
            )

        T = model.addVars(range(1, N), vtype=gurobipy.GRB.INTEGER, name="T")
        self.T = T

        threshold = self.theta
        # threshold = self.theta
        for n in tqdm(range(2, N), desc='两点之间的距离本身就超过阈值则直接剔除'):
            model.addConstr((T[n] == 1) >> (B[n] >= threshold))
            model.addConstr((T[n] == 0) >> (B[n] <= threshold - 1))

            for i in range(1, I):
                for j in range(1, I):
                    model.addConstr((T[n] == 1) >> (F[n - 1, i] + F[n, j] <= 1))

        self.model = model

        self.basic_model_inited = True
        print('Basic model initialized.')
        return model

    def add_adjust_varables(self, adjust_points, name_suffix):
        '''添加校正相关变量'''
        model = self.model
        N = self.N
        I = self.I
        Q = self.Q
        B = self.B
        F = self.F

        print(
            f'IsStepAjusted_{name_suffix}[n], 表示第n步走到的点是否是校验点, NotStepAjusted_{name_suffix}[n]表示取反'
        )
        IsStepAjusted = model.addVars(
            range(1, N), vtype=gurobipy.GRB.BINARY, name='IsStepAjusted_' + name_suffix
        )
        NotStepAjusted = model.addVars(
            range(1, N), vtype=gurobipy.GRB.BINARY, name='NotStepAjusted_' + name_suffix
        )
        for n in tqdm(range(1, N), desc=f'定义Fni和IsStepAjusted_{name_suffix}的关系'):
            for i in range(1, I):
                if i in adjust_points:
                    model.addConstr((F[n, i] == 1) >> (IsStepAjusted[n] == 1))
                else:
                    model.addConstr((F[n, i] == 1) >> (IsStepAjusted[n] == 0))

            model.addConstr((IsStepAjusted[n] == 1) >> (NotStepAjusted[n] == 0))
            model.addConstr((IsStepAjusted[n] == 0) >> (NotStepAjusted[n] == 1))

        # ==========================================================================================
        print(f'IsStepNCumX_{name_suffix}[n, x] 表示第n步是否累计了x次{name_suffix}误差没有校正')
        IsStepNCumX = model.addVars(
            range(1, N),
            range(1, I),
            vtype=gurobipy.GRB.BINARY,
            name="IsStepNCumX_" + name_suffix,
        )

        model.addConstrs(
            (Q[n] == 0) >> (IsStepAjusted[n] == 0)
            for n in tqdm(range(1, N), desc=f'Qn为0则IsStepAjusted_{name_suffix}[n]为0')
        )
        model.addConstrs(
            (Q[n] == 0) >> (IsStepNCumX.sum(n, '*') == 0)
            for n in tqdm(range(1, N), desc=f'Qn为0则IsStepNCumX_{name_suffix}[n, *]都为0')
        )
        # ==========================================================================================
        StepNCumB = model.addVars(
            range(1, N), vtype=gurobipy.GRB.INTEGER, name='StepNCumB_' + name_suffix
        )
        for n in tqdm(
            range(1, N), desc=f'StepNCumB{name_suffix}[n] 表示第n步累积了多少{name_suffix}误差'
        ):

            def sum_B(n1, n2):
                return gurobipy.quicksum(B[_] for _ in range(n1, n2 + 1))

            def add_x_constr(x):
                model.addConstr(
                    IsStepNCumX[n, x]
                    == gurobipy.and_(
                        [Q[n]]
                        + [NotStepAjusted[y] for y in range(n - x + 1, n)]
                        + ([IsStepAjusted[n - x]] if x < n else [])
                    )
                )
                model.addConstr(
                    (IsStepNCumX[n, x] == 1) >> (StepNCumB[n] == sum_B(n - x + 1, n))
                )

            if n == 1:
                model.addConstr(IsStepNCumX[1, 1] == 1)
                model.addConstr(StepNCumB[1] == sum_B(1, 1))
            else:
                for x in range(1, n + 1):
                    add_x_constr(x)
            for x in range(n + 1, N):
                model.addConstr(IsStepNCumX[n, x] == 0)
        return IsStepAjusted, IsStepNCumX, StepNCumB

    def add_adjust_constr(self):
        '''添加校正相关约束'''
        tdf = self.df.reset_index()
        vertical_points = tdf[tdf['校正点类型'] == 1].index.tolist()
        self.IsStepAjusted_V, self.IsStepNCumX_V, self.StepNCumB_V = self.add_adjust_varables(
            vertical_points, name_suffix='V'
        )

        horizontal_points = tdf[tdf['校正点类型'] == 0].index.tolist()
        self.IsStepAjusted_H, self.IsStepNCumX_H, self.StepNCumB_H = self.add_adjust_varables(
            horizontal_points, name_suffix='H'
        )

        for n in tqdm(range(1, self.N), desc=f'限制每步的累计误差'):
            self.model.addConstr(
                (self.IsStepAjusted_V[n] == 1) >> (self.StepNCumB_V[n] <= self.alpha1)
            )
            self.model.addConstr(
                (self.IsStepAjusted_V[n] == 1) >> (self.StepNCumB_H[n] <= self.alpha2)
            )
            self.model.addConstr(
                (self.IsStepAjusted_H[n] == 1) >> (self.StepNCumB_V[n] <= self.beta1)
            )
            self.model.addConstr(
                (self.IsStepAjusted_H[n] == 1) >> (self.StepNCumB_H[n] <= self.beta2)
            )

        for n in tqdm(range(1, self.N), desc='B点的累计误差小于theta'):
            self.model.addConstr(
                (self.F[n, self.I - 1] == 1) >> (self.StepNCumB_V[n] <= self.theta)
            )
            self.model.addConstr(
                (self.F[n, self.I - 1] == 1) >> (self.StepNCumB_H[n] <= self.theta)
            )

    def build_model(self):

        self.init_basic_model()

        self.add_adjust_constr()

        for n in tqdm(range(1, self.N), desc='加入一些冗余约束提高速度'):
            self.model.addConstr((self.Q[n] == 1) >> (self.C[n] >= 1))
            self.model.addConstr((self.Q[n] == 0) >> (self.B[n] == 0))
            self.model.addConstr(
                (self.Q[n] == 1) >> (self.B[n] >= 1)
            )  # Note: 当B设为整数时才能这么写

            self.model.addConstr(
                (self.Q[n] == 1)
                >> (self.IsStepNCumX_V.sum(n, '*') >= 1)  # Note: 当B设为整数时才能这么写
            )
            self.model.addConstr(
                (self.Q[n] == 1)
                >> (self.IsStepNCumX_H.sum(n, '*') >= 1)  # Note: 当B设为整数时才能这么写
            )

    def print_res(self):
        print(self.get_var_res('F', 1))

        print(self.get_var_res('Q', 1))

        print(self.get_var_res('B', not_val=0))

        print(self.get_var_res('T'))

        print(self.get_var_res('IsStepAjusted_V', 1))

        print(self.get_var_res('IsStepAjusted_H', 1))

    @property
    def routes_df(self):
        routes = self.get_var_res('C', not_val=0)
        routes_df = self.df.iloc[[0] + [round(_[1]) for _ in routes]]
        return routes_df

    @property
    def res_df(self):
        def get_cum_delta(VorH="V"):
            assert VorH in ("V", "H")

            actual_cum_delta = []
            for i in range(len(res_df)):
                if i == 0:
                    actual_cum_delta.append(0)
                    continue
                Bi = (
                    self.get_distance(
                        self.df.index.tolist().index(self.routes_df.index[i - 1]),
                        self.df.index.tolist().index(self.routes_df.index[i]),
                    )
                    * self.delta
                )
                if res_df.iloc[i - 1]["校正点类型"] == (VorH == "V"):
                    actual_cum_delta.append(Bi)
                else:
                    actual_cum_delta.append(Bi + actual_cum_delta[i - 1])
            return actual_cum_delta

        res_df = self.routes_df.copy()
        res_df["校正前垂直误差"] = get_cum_delta("V")
        res_df["校正前水平误差"] = get_cum_delta("H")
        res_df = res_df.rename_axis("校正点编号")[["校正前垂直误差", "校正前水平误差", "校正点类型"]]
        return res_df

    @property
    def total_distance(self):
        return sum(
            [
                self.get_distance(
                    self.df.index.tolist().index(self.routes_df.index[i - 1]),
                    self.df.index.tolist().index(self.routes_df.index[i]),
                )
                for i in range(1, len(self.routes_df))
            ]
        )

    def plot(self, *args, **kwargs):
        return self.get_init_fig(*args, **kwargs).add_trace(
            go.Scatter3d(
                x=self.routes_df["X坐标"],
                y=self.routes_df["Y坐标"],
                z=self.routes_df["Z坐标"],
                mode="lines",
                name="路径",
                line_width=2,
            )
        )

    def save(self, suffix):
        save_folder = Path("solutions") / f"dataset{self.which_dataset}_{suffix}"
        if not save_folder.is_dir():
            save_folder.mkdir(parents=True)
        self.model.write(str(save_folder / "model.mst"))
        self.model.write(str(save_folder / "model.sol"))

        self.df.to_csv(save_folder / "df.csv", encoding="gbk")
        self.routes_df.to_csv(save_folder / "routes_df.csv", encoding="gbk")
        self.res_df.to_csv(save_folder / "res_df.csv", encoding="gbk")

        def save_fig(fig, path):
            path.open("w").write(fig.update_layout(height=1000).to_html())

        save_fig(self.plot(True, "校正点类型"), Path(save_folder / "sample_type.html"))
        save_fig(self.plot(False, "校正点类型"), Path(save_folder / "full_type.html"))
        save_fig(
            self.plot(False, "in_subsample"), Path(save_folder / "full_subsample.html")
        )

        json.dump(
            {'total_distance': self.total_distance},
            Path(save_folder / "metrics.json").open('w'),
        )

        print("Saved to " + str(save_folder))


if __name__ == "__main__":
    s = Solver(subsample_how='barrel', subsample_rows=20, max_step=9)
    df = s.df
    N = s.N
    s.theta *= 10
    s.beta1 *= 10
    s.beta2 *= 10
    s.alpha1 *= 10
    s.alpha2 *= 10
    df

    s.build_model()
    model = s.model
    F, C, Q, B, T = s.F, s.C, s.Q, s.B, s.T

    # 创建目标函数
    model.setObjective(
        #     B.sum('*'),
        1,
        gurobipy.GRB.MINIMIZE,
    )

    # 执行线性规划模型
    # model.Params.MIPFocus = 3
    # model.Params.MultiObjPre = 2  # 激进的presolve
    # model.Params.Presolve = 2  # 激进的presolve
    # model.Params.Heuristics = 0
    # model.Params.TimeLimit = 5 * 60
    model.optimize()
    print("Obj:", model.objVal)

    s.print_res()
    # s.plot(True).show()
    s.plot(False).show()
