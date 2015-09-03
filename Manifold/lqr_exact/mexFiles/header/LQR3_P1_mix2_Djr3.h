#ifndef LQR3_P1_mix2_Djr3_H
#define LQR3_P1_mix2_Djr3_H

double LQR3_P1_mix2_Djr3(double* tv, int tdim, double* param, int pdim) {

double t1 = tv[0], t2 = tv[1];

double r1 = param[0];
double r2 = param[1];
double r3 = param[2];
double r4 = param[3];
double r5 = param[4];
double r6 = param[5];
double r7 = param[6];
double r8 = param[7];
double r9 = param[8];

double t4 = r4-3.338299811;
double t5 = t1*t4;
double t6 = r5*t2;
double t7 = t1*t1;
double t8 = t2*t2;
double t11 = r4*t7;
double t12 = r5*t8;
double t13 = r6*t1*t2;
double t9 = t5+t6-t11-t12-t13+1.151035476;
double t10 = exp(t9);
double t14 = t10+1.0;
double t15 = 1.0/t14;
double t16 = 1.0/(t14*t14);
double t17 = t15*1.8E1;
double t19 = t16*9.0;
double t18 = t17-t19+1.0;
double t20 = 1.0/t18;
double t21 = t15*t20*2.18E2;
double t22 = t16*(1.0/1.0E1);
double t23 = t22+9.0/1.0E1;
double t24 = t17-1.8E1;
double t25 = 1.0/(t18*t18);
double t26 = t23*t24*t25*1.09E3;
double t27 = t21+t26;
double t28 = r7+3.338299811;
double t29 = t1*t28;
double t30 = r8+3.338299811;
double t31 = t2*t30;
double t34 = r7*t7;
double t35 = r8*t8;
double t36 = r9*t1*t2;
double t32 = t29+t31-t34-t35-t36-2.187264336;
double t33 = exp(t32);
double t37 = t33+1.0;
double t38 = 1.0/t37;
double t39 = 1.0/(t37*t37);
double t40 = t38*1.8E1;
double t63 = t39*9.0;
double t41 = t40-t63+1.0;
double t42 = r2-3.338299811;
double t43 = t2*t42;
double t44 = r1*t1;
double t47 = r1*t7;
double t48 = r2*t8;
double t49 = r3*t1*t2;
double t45 = t43+t44-t47-t48-t49+1.151035476;
double t46 = exp(t45);
double t50 = t46+1.0;
double t51 = 1.0/t50;
double t52 = 1.0/(t50*t50);
double t53 = t51*1.8E1;
double t68 = t52*9.0;
double t54 = t53-t68+1.0;
double t55 = t15*t20*1.962E3;
double t56 = t16*(9.0/1.0E1);
double t57 = t56+1.0/1.0E1;
double t58 = t24*t25*t57*1.09E3;
double t59 = t55+t58;
double t60 = r4*t1*2.0;
double t61 = r6*t2;
double t62 = -r4+t60+t61+3.338299811;
double t64 = 1.0/t41;
double t65 = t40-1.8E1;
double t66 = 1.0/(t41*t41);
double t87 = r7*t1*2.0;
double t88 = r9*t2;
double t67 = r7-t87-t88+3.338299811;
double t69 = 1.0/t54;
double t70 = t51*t69*1.962E3;
double t71 = t52*(9.0/1.0E1);
double t72 = t71+1.0/1.0E1;
double t73 = t53-1.8E1;
double t74 = 1.0/(t54*t54);
double t75 = t72*t73*t74*1.09E3;
double t76 = t70+t75;
double t77 = r1*t1*2.0;
double t78 = r3*t2;
double t79 = -r1+t77+t78;
double t80 = t46*t52*t76*t79;
double t81 = t10*t16*t59*t62;
double t82 = t38*t64*1.962E3;
double t83 = t39*(9.0/1.0E1);
double t84 = t83+1.0/1.0E1;
double t85 = t65*t66*t84*1.09E3;
double t86 = t82+t85;
double t89 = t38*t64*2.18E2;
double t90 = t39*(1.0/1.0E1);
double t91 = t90+9.0/1.0E1;
double t92 = t65*t66*t91*1.09E3;
double t93 = t89+t92;
double t105 = t33*t39*t67*t93;
double t94 = t80+t81-t105;
double t95 = t10*t16*t27*t62;
double t97 = t33*t39*t67*t86;
double t96 = t80+t95-t97;
double t98 = t51*t69*2.18E2;
double t99 = t52*(1.0/1.0E1);
double t100 = t99+9.0/1.0E1;
double t101 = t73*t74*t100*1.09E3;
double t102 = t98+t101;
double t103 = t46*t52*t79*t102;
double t104 = t81-t97+t103;
double t106 = r2*t2*2.0;
double t107 = r3*t1;
double t108 = -r2+t106+t107+3.338299811;
double t115 = r8*t2*2.0;
double t116 = r9*t1;
double t109 = r8-t115-t116+3.338299811;
double t110 = r5*t2*2.0;
double t111 = r6*t1;
double t112 = -r5+t110+t111;
double t113 = t10*t16*t59*t112;
double t114 = t46*t52*t76*t108;
double t123 = t33*t39*t93*t109;
double t117 = t113+t114-t123;
double t118 = t46*t52*t102*t108;
double t120 = t33*t39*t86*t109;
double t119 = t113+t118-t120;
double t121 = t10*t16*t27*t112;
double t122 = t114-t120+t121;
double t124 = t102*t104;
double t125 = t76*t96;
double t126 = t76*t94;
double t127 = t124+t125+t126;
double t128 = t93*t94;
double t129 = t86*t96;
double t130 = t86*t104;
double t131 = t128+t129+t130;
double t132 = t27*t96;
double t133 = t59*t94;
double t134 = t27*t122;
double t135 = t59*t119;
double t136 = t59*t117;
double t137 = t134+t135+t136;
double t138 = t93*t117;
double t139 = t86*t119;
double t140 = t86*t122;
double t141 = t138+t139+t140;
double t142 = t102*t119;
double t143 = t76*t117;
double t144 = 1.0/AUp1;
double t145 = t69*t100*1.09E3;
double t146 = t20*t57*1.09E3;
double t147 = t64*t84*1.09E3;
double t148 = t145+t146+t147+1.9E1;
double t149 = t144*t148;
double t150 = t149-1.0;
double t151 = 1.0/(t50*t50*t50);
double t152 = 1.0/AUp2;
double t153 = t69*t72*1.09E3;
double t154 = t20*t23*1.09E3;
double t155 = t147+t153+t154+1.9E1;
double t156 = t152*t155;
double t157 = t156-1.0;
double t158 = t1*t2*t46*t52*1.8E1;
double t165 = t1*t2*t46*t151*1.8E1;
double t159 = t158-t165;
double t160 = 1.0/AUp3;
double t161 = t64*t91*1.09E3;
double t162 = t146+t153+t161+1.9E1;
double t163 = t160*t162;
double t164 = t163-1.0;
double t166 = t72*t74*t159*1.09E3;
double t189 = t1*t2*t46*t69*t151*1.962E3;
double t167 = t166-t189;
double t174 = 1.0/Up1;
double t175 = t148*t174;
double t176 = t175-1.0;
double t168 = fabs(t176);
double t178 = 1.0/Up2;
double t179 = t155*t178;
double t180 = t179-1.0;
double t169 = fabs(t180);
double t182 = 1.0/Up3;
double t183 = t162*t182;
double t184 = t183-1.0;
double t170 = fabs(t184);
double t171 = fabs(t150);
double t172 = fabs(t157);
double t173 = fabs(t164);
double t177 = t168*t168;
double t181 = t169*t169;
double t185 = t170*t170;
double t186 = t177+t181+t185;
double t187 = t74*t100*t159*1.09E3;
double t188 = t187-t1*t2*t46*t69*t151*2.18E2;
double t190 = t59*t104;
double t191 = t132+t133+t190;
double t192 = t76*t122;
double t193 = t142+t143+t192;
double t194 = t10*t16*t62*t191;
double t195 = t46*t52*t79*t127;
double t243 = t33*t39*t67*t131;
double t196 = t194+t195-t243;
double t197 = t46*t52*t108*t193;
double t198 = t10*t16*t112*t137;
double t254 = t33*t39*t109*t141;
double t199 = t197+t198-t254;
double t200 = t196*t199;
double t201 = t46*t52*t108*t127;
double t202 = t10*t16*t112*t191;
double t269 = t33*t39*t109*t131;
double t203 = t201+t202-t269;
double t204 = t10*t16*t62*t137;
double t205 = t46*t52*t79*t193;
double t213 = t33*t39*t67*t141;
double t206 = t204+t205-t213;
double t207 = t200-t203*t206;
double t208 = t171*t171;
double t209 = t172*t172;
double t210 = t173*t173;
double t211 = t208+t209+t210;
double t212 = 1.0/t186;
double t214 = 1.0/(t54*t54*t54);
double t215 = t2*t42*2.0;
double t224 = r1*t7*2.0;
double t225 = r2*t8*2.0;
double t226 = r3*t1*t2*2.0;
double t216 = t77+t215-t224-t225-t226+2.302070952;
double t217 = exp(t216);
double t218 = t1*t2*t46*t52*t69*1.962E3;
double t219 = t1*t2*t46*t52*t72*t74*1.962E4;
double t220 = t1*t2*t46*t73*t74*t151*1.962E3;
double t233 = t51*t74*t159*1.962E3;
double t234 = t72*t73*t159*t214*2.18E3;
double t221 = t218+t219+t220-t233-t234;
double t222 = t46*t52*t79*t221;
double t223 = t2*t46*t52*t76;
double t227 = t1*t2*t76*t79*t151*t217*2.0;
double t241 = t1*t2*t46*t52*t76*t79;
double t228 = t222+t223+t227-t241;
double t229 = t1*t2*t46*t52*t69*2.18E2;
double t230 = t1*t2*t46*t52*t74*t100*1.962E4;
double t231 = t1*t2*t46*t73*t74*t151*2.18E2;
double t235 = t51*t74*t159*2.18E2;
double t236 = t73*t100*t159*t214*2.18E3;
double t232 = t229+t230+t231-t235-t236;
double t237 = t46*t52*t79*t232;
double t238 = t2*t46*t52*t102;
double t239 = t1*t2*t79*t102*t151*t217*2.0;
double t242 = t1*t2*t46*t52*t79*t102;
double t240 = t237+t238+t239-t242;
double t244 = t1*t46*t52*t76;
double t245 = t46*t52*t108*t221;
double t246 = t1*t2*t76*t108*t151*t217*2.0;
double t252 = t1*t2*t46*t52*t76*t108;
double t247 = t244+t245+t246-t252;
double t248 = t1*t46*t52*t102;
double t249 = t46*t52*t108*t232;
double t250 = t1*t2*t102*t108*t151*t217*2.0;
double t253 = t1*t2*t46*t52*t102*t108;
double t251 = t248+t249+t250-t253;
double t255 = t59*t240;
double t256 = t27*t228;
double t257 = t59*t228;
double t258 = t255+t256+t257;
double t259 = t86*t240;
double t260 = t93*t228;
double t261 = t86*t228;
double t262 = t259+t260+t261;
double t263 = t104*t232;
double t264 = t96*t221;
double t265 = t94*t221;
double t266 = t102*t240;
double t267 = t76*t228*2.0;
double t268 = t263+t264+t265+t266+t267;
double t270 = t119*t232;
double t271 = t117*t221;
double t272 = t122*t221;
double t273 = t102*t251;
double t274 = t76*t247*2.0;
double t275 = t270+t271+t272+t273+t274;
double t276 = t59*t251;
double t277 = t27*t247;
double t278 = t59*t247;
double t279 = t276+t277+t278;
double t280 = t86*t251;
double t281 = t93*t247;
double t282 = t86*t247;
double t283 = t280+t281+t282;
double t0 = -sqrt(t207)*(beta1*t212*(t152*t167*t172*((t157/fabs(t157)))*2.0+t144*t171*t188*((t150/fabs(t150)))*2.0+t160*t167*t173*((t164/fabs(t164)))*2.0)-beta1*1.0/(t186*t186)*t211*(t167*t169*t178*((t180/fabs(t180)))*2.0+t167*t170*t182*((t184/fabs(t184)))*2.0+t168*t174*t188*((t176/fabs(t176)))*2.0))-1.0/sqrt(t207)*(beta2-beta1*t211*t212)*(t199*(t2*t46*t52*t127+t10*t16*t62*t258-t33*t39*t67*t262+t46*t52*t79*t268-t1*t2*t46*t52*t79*t127+t1*t2*t79*t127*t151*t217*2.0)-t206*(t1*t46*t52*t127+t10*t16*t112*t258-t33*t39*t109*t262+t46*t52*t108*t268-t1*t2*t46*t52*t108*t127+t1*t2*t108*t127*t151*t217*2.0)-t203*(t2*t46*t52*t193+t10*t16*t62*t279-t33*t39*t67*t283+t46*t52*t79*t275-t1*t2*t46*t52*t79*t193+t1*t2*t79*t151*t193*t217*2.0)+t196*(t1*t46*t52*t193+t10*t16*t112*t279-t33*t39*t109*t283+t46*t52*t108*t275-t1*t2*t46*t52*t108*t193+t1*t2*t108*t151*t193*t217*2.0))*(1.0/2.0);

return t0;
}

#endif