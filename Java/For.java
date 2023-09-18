package hello;

public class For {
	public static void main(String[] args) {
		//(1)標準寫法
		for(int i=0;i<10;i++) {
			if(i==5) {
				continue;
			}
			System.out.println(i);
		}
		//(2)表達式一省略，但在外部聲明
		int i2 = 0;
		for(;i2<10;i2++) {
				System.out.println(i2);
		}
		/**(3)表達式二省略，死循環
		for(int i3=0;;i3++) {
			System.out.println(i3);
			}
			*/
		//(4)表達式三省略，要聲明在內部
		for(int i4=0;i4<10;) {
				System.out.println(i4);
				i4++;
		}
		//(5)表達式1、3省略，但要聲明再循環的外部及內部
		int i5 = 0;
		for(;i5<10;) {
			System.out.println(i5);
			i5++;
		}
		/**(6)三個表達式都省略，死循環
		 *for(;;)
		 */
	}
}
